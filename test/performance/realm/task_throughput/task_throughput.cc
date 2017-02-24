/* Copyright 2017 Stanford University
 * Copyright 2017 Los Alamos National Laboratory 
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <unistd.h>

#include <time.h>

#include <realm/realm.h>
#include <realm/cmdline.h>

using namespace Realm;
using namespace LegionRuntime::Accessor;

namespace TestConfig {
  int tasks_per_processor = 256;
  int launching_processors = 1;
  int task_argument_size = 0;
  bool remote_tasks = false;
  bool with_profiling = false;
};

// TASK IDs
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0, 
  TASK_LAUNCHER,
  DUMMY_TASK,
  PROFILER_TASK,
};

Logger log_app("app");

// dummy tasks are marked as either "first", "middle", or "last", and prioritized the same way
enum TaskOrder {
  FIRST_TASK = 10,
  MIDDLE_TASK = 9,
  LAST_TASK = 8,
};

struct TestTaskArgs {
  int which_task;
  RegionInstance instance;
  Barrier finish_barrier;
};

struct TestTaskData {
  int first_count;
  int last_count;
  int total_tasks;
  double start_time;
};

void dummy_task(const void *args, size_t arglen, 
		const void *userdata, size_t userlen, Processor p)
{
  const TestTaskArgs& ta = *(const TestTaskArgs *)args;
  int task_type = ta.which_task;
  // quick out for most tasks
  if(task_type == MIDDLE_TASK) return;

  RegionAccessor<AccessorType::Affine<1>, TestTaskData> ra = ta.instance.get_accessor().typeify<TestTaskData>().convert<AccessorType::Affine<1> >();
  TestTaskData& mydata = ra[0];

  if(task_type == FIRST_TASK) {
    double t = Clock::current_time();
    log_app.debug() << "first task on " << p << ": " << t;
    assert(mydata.last_count == 0);
    if(mydata.first_count == 0)
      mydata.start_time = t;
    mydata.first_count++;
  }
  if(task_type == LAST_TASK) {
    double t = Clock::current_time();
    log_app.debug() << "last task on " << p << ": " << t;
    assert(mydata.last_count < mydata.first_count);
    mydata.last_count++;
    if(mydata.last_count == mydata.first_count) {
      double elapsed = t - mydata.start_time;
      double per_task = elapsed / (mydata.last_count * 
				   TestConfig::tasks_per_processor);
      log_app.print() << "tasks complete on " << p << ": " 
		      << (1e6 * per_task) << " us/task, "
		      << (1.0 / per_task) << " tasks/s";
      ta.finish_barrier.arrive();
    }
  }
}

void profiler_task(const void *args, size_t arglen, 
		   const void *userdata, size_t userlen, Processor p)
{
  // do nothing with the data
}

struct LauncherArgs {
  Barrier start_barrier;
  Barrier finish_barrier;
  std::map<Processor, RegionInstance> instances;
};

template<typename S>
bool serdez(S& s, const LauncherArgs& t)
{
  return (s & t.start_barrier) && (s & t.finish_barrier) && (s & t.instances);
}


void task_launcher(const void *args, size_t arglen, 
		   const void *userdata, size_t userlen, Processor p)
{
  Serialization::FixedBufferDeserializer fbd(args, arglen);
  LauncherArgs la;
  bool ok = fbd >> la;
  assert(ok);

  Machine::ProcessorQuery pq(Machine::get_machine());
  if(!TestConfig::remote_tasks)
    pq = pq.same_address_space_as(p);

  ProfilingRequestSet prs;
  if(TestConfig::with_profiling) {
    using namespace Realm::ProfilingMeasurements;
    prs.add_request(p, PROFILER_TASK).add_measurement<OperationTimeline>();
  }

  // allocate some space for our test arguments
  int argsize = sizeof(TestTaskArgs);
  if(TestConfig::task_argument_size > argsize)
    argsize = TestConfig::task_argument_size;
  TestTaskArgs *tta = (TestTaskArgs *)(alloca(argsize));

  // time how long this takes us
  double t1 = Clock::current_time();
  int total_tasks = 0;
  for(int i = 0; i < TestConfig::tasks_per_processor; i++) {
    int which = ((i == 0) ? FIRST_TASK :
		 (i == (TestConfig::tasks_per_processor - 1)) ? LAST_TASK :
		 MIDDLE_TASK);
    for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it) {
      tta->which_task = which;
      tta->instance = la.instances[*it];
      assert(tta->instance.exists());
      tta->finish_barrier = la.finish_barrier;
      (*it).spawn(DUMMY_TASK, tta, argsize, prs, la.start_barrier, which);
      total_tasks++;
    }
  }
  double t2 = Clock::current_time();

  double spawn_rate = total_tasks / (t2 - t1);
  log_app.print() << "spawn rate on " << p << ": " << spawn_rate << " tasks/s";

  // we're all done - we can arrive at the start barrier and then finish this task
  la.start_barrier.arrive();
}

void top_level_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  LauncherArgs launch_args;

  // go through all processors and organize by address space
  std::map<AddressSpace, std::vector<Processor> > all_procs;
  std::map<AddressSpace, std::vector<Processor> > loc_procs;
  int total_procs = 0;
  {
    std::set<Event> events;
    Machine::ProcessorQuery pq(Machine::get_machine());
    for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it) {
      Processor p = *it;
      AddressSpace a = p.address_space();
      all_procs[a].push_back(p);
      if(p.kind() == Processor::LOC_PROC)
	loc_procs[a].push_back(p);
      total_procs++;

      Memory m = Machine::MemoryQuery(Machine::get_machine())
	.has_affinity_to(p)
	.first();
      assert(m.exists());
      Domain d = Domain::from_rect<1>(Rect<1>(0, 0));
      RegionInstance i = d.create_instance(m, sizeof(TestTaskData));
      launch_args.instances[p] = i;
      {
	std::vector<Domain::CopySrcDstField> dsts(1);
	dsts[0].inst = i;
	dsts[0].offset = 0;
	dsts[0].size = sizeof(TestTaskData);
	TestTaskData ival;
	ival.first_count = 0;
	ival.last_count = 0;
	ival.start_time = 0;
	d.fill(dsts, &ival, sizeof(ival)).wait();
      }
    }
  }

  // two barriers will coordinate the running of the test tasks
  // 1) one triggered by each of the launcher tasks that starts the tasks running
  // 2) one triggered by each processor when all task launches have been seen
  launch_args.start_barrier = Barrier::create_barrier(all_procs.size() * 
						      TestConfig::launching_processors);
  launch_args.finish_barrier = Barrier::create_barrier(total_procs);

  // serialize the launcher args
  void *args_data;
  size_t args_size;
  {
    Serialization::DynamicBufferSerializer dbs(256);
    bool ok = dbs << launch_args;
    assert(ok);
    args_size = dbs.bytes_used();
    args_data = dbs.detach_buffer();
  }    

  // spawn launcher tasks in each address space
  for(std::map<AddressSpace, std::vector<Processor> >::const_iterator it = all_procs.begin();
      it != all_procs.end();
      ++it) {
    const std::vector<Processor>& lp = loc_procs[it->first];
    assert(lp.size() >= (size_t)(TestConfig::launching_processors));
    for(int i = 0; i < TestConfig::launching_processors; i++) {
      Processor p = lp[i];
      
      // no need to grab the finish event - we wait indirectly via the barrier
      p.spawn(TASK_LAUNCHER, args_data, args_size);
    }
  }

  // all done - wait for everything to finish via the finish_barrier
  launch_args.finish_barrier.wait();
}

int main(int argc, char **argv)
{
  Runtime r;

  bool ok = r.init(&argc, &argv);
  assert(ok);

  CommandLineParser cp;
  cp.add_option_int("-tpp", TestConfig::tasks_per_processor)
    .add_option_int("-lp", TestConfig::launching_processors)
    .add_option_int("-args", TestConfig::task_argument_size)
    .add_option_bool("-remote", TestConfig::remote_tasks)
    .add_option_bool("-prof", TestConfig::with_profiling);
  ok = cp.parse_command_line(argc, (const char **)argv);
  assert(ok);

  r.register_task(TOP_LEVEL_TASK, top_level_task);
  r.register_task(TASK_LAUNCHER, task_launcher);
  r.register_task(DUMMY_TASK, dummy_task);
  r.register_task(PROFILER_TASK, profiler_task);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = r.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // request shutdown once that task is complete
  r.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  r.wait_for_shutdown();
  
  return 0;
}
