/* Copyright 2017 Stanford University
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
#include <cassert>
#include <cstdlib>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>
#include "legion.h"

#include "default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

//#define USE_DEFAULT

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  DAXPY_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_Z,
};

enum {
  SUBREGION_TUNABLE,
};

enum {
  PARTITIONING_MAPPER_ID = 1,
};

typedef enum {
  TASK_STEAL_REQUEST = 1,
  TASK_STEAL_ACK,
  TASK_STEAL_CONTINUE,
  POOL_POOL_FORWARD_STEAL_SUCCESS,
  POOL_WORKER_STEAL_NACK,
  POOL_WORKER_WAKEUP,
} MessageType;


typedef struct task_steal_request_s{
  Processor target_proc;
  unsigned num_tasks;
} task_steal_request_t;


typedef struct task_profiling_s {
  long long duration;
  bool      is_recursive_task;
}task_profiling_t;

double get_cur_time() {
  struct timeval   tv;
  struct timezone  tz;
  double cur_time;
  
  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;
  
  return cur_time;
} 


typedef struct recursive_task_args_s {
  bool recursiveable;
  bool is_recursive_task;
}recursive_task_args_t;

class RecursiveTaskArgument {
private:
  recursive_task_args_t recursive_task_args;
  void*                 args;
  size_t                arglen;
public:
  RecursiveTaskArgument(const void *usr_args, size_t user_arglen, 
                        bool recursiveable, bool is_recursive_task);
  ~RecursiveTaskArgument();
public:
  static void*  get_usr_args(void* task_args);
  static size_t get_usr_arglen(size_t task_arglen);
  static bool   is_task_recursiveable(const Task *task);
  static bool   is_task_recursive_task(const Task *task);
  static void   set_task_recursiveable(const Task *task);
  static void   set_task_is_recursive_task(const Task *task);
public:
  void*   get_args(void);
  size_t  get_arglen(void);
private:
  void    pack_task_args(const void *usr_args, size_t user_arglen);            
};

RecursiveTaskArgument::RecursiveTaskArgument(const void *usr_args, 
                                             size_t user_arglen, 
                                             bool recursiveable, 
                                             bool is_recursive_task)
{
  recursive_task_args.recursiveable = recursiveable;
  recursive_task_args.is_recursive_task = is_recursive_task;
  pack_task_args(usr_args, user_arglen);
}

RecursiveTaskArgument::~RecursiveTaskArgument()
{
  free(args);
  args = NULL;
}

void* RecursiveTaskArgument::get_usr_args(void* task_args)
{
  if (task_args == NULL) {
    return NULL;
  }
  unsigned char *ptr = (unsigned char*)task_args;
  ptr += sizeof(recursive_task_args_t);
  return ptr;
}

size_t RecursiveTaskArgument::get_usr_arglen(size_t task_arglen)
{
  if (task_arglen == 0) {
    return 0;
  }
  size_t usr_arglen = task_arglen - sizeof(recursive_task_args_t);
  assert(usr_arglen >= 0);
  return usr_arglen;
}

bool RecursiveTaskArgument::is_task_recursiveable(const Task *task)
{
  recursive_task_args_t *recursive_args = (recursive_task_args_t*)task->args;
  if (recursive_args == NULL || task->task_id == 0) {
    return false;
  }
  return recursive_args->recursiveable;
}

bool RecursiveTaskArgument::is_task_recursive_task(const Task *task)
{
  recursive_task_args_t *recursive_args = (recursive_task_args_t*)task->args;
  if (recursive_args == NULL || task->task_id == 0) {
    return false;
  }
  return recursive_args->is_recursive_task;
}

void RecursiveTaskArgument::set_task_recursiveable(const Task *task)
{
  recursive_task_args_t *recursive_args = (recursive_task_args_t*)task->args;
  assert(recursive_args != NULL);
  recursive_args->recursiveable = true;
}

void RecursiveTaskArgument::set_task_is_recursive_task(const Task *task)
{
  recursive_task_args_t *recursive_args = (recursive_task_args_t*)task->args;
  assert(recursive_args != NULL);
  recursive_args->is_recursive_task = true;
}

void* RecursiveTaskArgument::get_args(void)
{
  return args;
}

size_t RecursiveTaskArgument:: get_arglen(void)
{
  return arglen;
}

void RecursiveTaskArgument::pack_task_args(const void *usr_args, 
                                           size_t usr_arglen)
{
  args = (void*)malloc(sizeof(recursive_task_args_t) + usr_arglen);
  arglen = sizeof(recursive_task_args_t) + usr_arglen;
  unsigned char *ptr = (unsigned char*)args;
  memcpy(ptr, &recursive_task_args, sizeof(recursive_task_args_t));
  ptr += sizeof(recursive_task_args_t);
  memcpy(ptr, (void*)usr_args, usr_arglen);
}

class RecursiveTaskMapperShared {
public:
  RecursiveTaskMapperShared();
public:
  std::map<TaskID, task_profiling_t> task_profiling_history;
  std::map<TaskID, bool> task_use_recursive;
  double task_slowdown_allowance;
  int max_recursive_tasks_to_schedule;
  int recursive_tasks_scheduled;
};

RecursiveTaskMapperShared::RecursiveTaskMapperShared()
{
  task_slowdown_allowance = 2;
  max_recursive_tasks_to_schedule = 1;
  recursive_tasks_scheduled = 0;
}

class AdaptiveMapper : public DefaultMapper {
public:
  AdaptiveMapper(Machine machine, 
      Runtime *rt, Processor local, RecursiveTaskMapperShared *shared);
public:
  virtual void handle_message(const MapperContext  ctx,
                              const MapperMessage& message);
                              
  virtual void select_task_options(const MapperContext ctx,
                                   const Task&         task,
                                         TaskOptions&  output);
                                 
  virtual void slice_task(const MapperContext      ctx,
                          const Task&              task,
                          const SliceTaskInput&    input,
                                SliceTaskOutput&   output);
  
  virtual void select_tasks_to_map(const MapperContext          ctx,
                                   const SelectMappingInput&    input,
                                         SelectMappingOutput&   output);
                                         
  virtual void map_task(const MapperContext   ctx,
                        const Task&           task,
                        const MapTaskInput&   input,
                              MapTaskOutput&  output);
                              
  virtual void select_steal_targets(const MapperContext         ctx,
                                    const SelectStealingInput&  input,
                                          SelectStealingOutput& output);
       
  virtual void permit_steal_request(const MapperContext         ctx,
                                    const StealRequestInput&    input,
                                          StealRequestOutput&   output);
                                          
  virtual void report_profiling(const MapperContext      ctx,
				                        const Task&              task,
				                        const TaskProfilingInfo& input);
protected:
  Processor select_processor_by_id(Processor::Kind kind, const unsigned int id);
  
  Processor select_stealable_processor(Processor::Kind kind);
  
  const std::map<VariantID,Processor::Kind>& find_task_variants(
                                            MapperContext ctx, TaskID task_id);
  
  void trigger_select_tasks_to_map(const MapperContext ctx);
  
  long generate_random_integer(void) const 
        { return default_generate_random_integer(); }
  
  void default_map_task(const MapperContext   ctx,
                        const Task&           task,
                        const MapTaskInput&   input,
                              MapTaskOutput&  output);   

private:
  std::map<TaskID,std::map<VariantID,
                            Processor::Kind> > variant_processor_kinds;

  MapperEvent defer_select_tasks_to_map;
  int recursive_tasks_scheduled;
  double task_slowdown_allowance;
  int max_recursive_tasks_to_schedule;
  int num_tasks_per_slice;
  std::map<TaskID, task_profiling_t> task_profiling_history;
  std::map<TaskID, bool> task_use_recursive;
  std::set<Processor> task_stealable_processor_list;
  std::deque<task_steal_request_t> task_steal_request_queue;
  bool select_tasks_to_map_local;
  bool slow_down_mapper;
  
};

static LegionRuntime::Logger::Category log_adapt_mapper("adapt_mapper");


void mapper_registration(Machine machine, Runtime *rt,
                          const std::set<Processor> &local_procs)
{
  RecursiveTaskMapperShared *mapper_shared = new RecursiveTaskMapperShared();
  printf("local procs size %d\n", local_procs.size());
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(
        new AdaptiveMapper(machine, rt, *it, mapper_shared), *it);
  }
}

//--------------------------------------------------------------------------
AdaptiveMapper::AdaptiveMapper(Machine m, 
                                     Runtime *rt, Processor p, 
                                     RecursiveTaskMapperShared *mapper_shared_variables)
  : DefaultMapper(rt->get_mapper_runtime(), m, p) // pass arguments through to DefaultMapper 
//--------------------------------------------------------------------------
{
  // init
  task_slowdown_allowance = 2;
  max_recursive_tasks_to_schedule = 1;
  recursive_tasks_scheduled = 0;
  select_tasks_to_map_local = true;
  num_tasks_per_slice = 1;
  task_stealable_processor_list.clear();
  slow_down_mapper = false;
  
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  
  // Recall that we create one mapper for every processor.  We
  // only want to print out this information one time, so only
  // do it if we are the mapper for the first processor in the
  // list of all processors in the machine.
  if (all_procs.begin()->id + 1 == local_proc.id)
  {
    // Print out how many processors there are and each
    // of their kinds.
    printf("There are %zd processors:\n", all_procs.size());
    for (std::set<Processor>::const_iterator it = all_procs.begin();
          it != all_procs.end(); it++)
    {
      // For every processor there is an associated kind
      Processor::Kind kind = it->kind();
      switch (kind)
      {
        // Latency-optimized cores (LOCs) are CPUs
        case Processor::LOC_PROC:
          {
            printf("  Processor ID " IDFMT " is CPU\n", it->id); 
            break;
          }
        // Throughput-optimized cores (TOCs) are GPUs
        case Processor::TOC_PROC:
          {
            printf("  Processor ID " IDFMT " is GPU\n", it->id);
            break;
          }
        // Processor for doing I/O
        case Processor::IO_PROC:
          {
            printf("  Processor ID " IDFMT " is I/O Proc\n", it->id);
            break;
          }
        // Utility processors are helper processors for
        // running Legion runtime meta-level tasks and 
        // should not be used for running application tasks
        case Processor::UTIL_PROC:
          {
            printf("  Processor ID " IDFMT " is utility\n", it->id);
            break;
          }
        default:
          assert(false);
      }
    }
    // We can also get the list of all the memories available
    // on the target architecture and print out their info.
    std::set<Memory> all_mems;
    machine.get_all_memories(all_mems);
    printf("There are %zd memories:\n", all_mems.size());
    for (std::set<Memory>::const_iterator it = all_mems.begin();
          it != all_mems.end(); it++)
    {
      Memory::Kind kind = it->kind();
      size_t memory_size_in_kb = it->capacity() >> 10;
      switch (kind)
      {
        // RDMA addressable memory when running with GASNet
        case Memory::GLOBAL_MEM:
          {
            printf("  GASNet Global Memory ID " IDFMT " has %zd KB\n", 
                    it->id, memory_size_in_kb);
            break;
          }
        // DRAM on a single node
        case Memory::SYSTEM_MEM:
          {
            printf("  System Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Pinned memory on a single node
        case Memory::REGDMA_MEM:
          {
            printf("  Pinned Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // A memory associated with a single socket
        case Memory::SOCKET_MEM:
          {
            printf("  Socket Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Zero-copy memory betweeen CPU DRAM and
        // all GPUs on a single node
        case Memory::Z_COPY_MEM:
          {
            printf("  Zero-Copy Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // GPU framebuffer memory for a single GPU
        case Memory::GPU_FB_MEM:
          {
            printf("  GPU Frame Buffer Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Disk memory on a single node
        case Memory::DISK_MEM:
          {
            printf("  Disk Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // HDF framebuffer memory for a single GPU
        case Memory::HDF_MEM:
          {
            printf("  HDF Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // File memory on a single node
        case Memory::FILE_MEM:
          {
            printf("  File Memory ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Block of memory sized for L3 cache
        case Memory::LEVEL3_CACHE:
          {
            printf("  Level 3 Cache ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Block of memory sized for L2 cache
        case Memory::LEVEL2_CACHE:
          {
            printf("  Level 2 Cache ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        // Block of memory sized for L1 cache
        case Memory::LEVEL1_CACHE:
          {
            printf("  Level 1 Cache ID " IDFMT " has %zd KB\n",
                    it->id, memory_size_in_kb);
            break;
          }
        default:
          assert(false);
      }
    }

    std::set<Memory> vis_mems;
    machine.get_visible_memories(local_proc, vis_mems);
    printf("There are %zd memories visible from processor " IDFMT "\n",
            vis_mems.size(), local_proc.id);
    for (std::set<Memory>::const_iterator it = vis_mems.begin();
          it != vis_mems.end(); it++)
    {
      // Edges between nodes are called affinities in the
      // machine model.  Affinities also come with approximate
      // indications of the latency and bandwidth between the 
      // two nodes.  Right now these are unit-less measurements,
      // but our plan is to teach the Legion runtime to profile
      // these values on start-up to give them real values
      // and further increase the portability of Legion applications.
      std::vector<ProcessorMemoryAffinity> affinities;
      int results = 
        machine.get_proc_mem_affinity(affinities, local_proc, *it);
      // We should only have found 1 results since we
      // explicitly specified both values.
      assert(results == 1);
      printf("  Memory " IDFMT " has bandwidth %d and latency %d\n",
              it->id, affinities[0].bandwidth, affinities[0].latency);
    }
  }
}

//--------------------------------------------------------------------------
void AdaptiveMapper::handle_message(const MapperContext ctx,
                                 const MapperMessage& message)
//--------------------------------------------------------------------------
{
  switch(message.kind) {
    case TASK_STEAL_REQUEST:
    {
      if (message.sender != local_proc) {
        // add sender to stealable processor list
        std::set<Processor>::iterator it;
        it = task_stealable_processor_list.find(message.sender);
        if (it == task_stealable_processor_list.end()) {
          task_stealable_processor_list.insert(message.sender);
        }
        task_steal_request_t request = {local_proc, 2};
        runtime->send_message(ctx, message.sender, &request, sizeof(task_steal_request_t), TASK_STEAL_ACK);
        log_adapt_mapper.debug("%s, local_proc: %llx, received a message from proc %llx, STEAL", __FUNCTION__, local_proc.id, message.sender.id);
      }
      break;
    }
    case TASK_STEAL_ACK:
    {
      select_tasks_to_map_local = false;
      task_steal_request_t request = *(task_steal_request_t*)message.message;
      task_steal_request_queue.push_back(request);
      trigger_select_tasks_to_map(ctx);
      log_adapt_mapper.debug("%s, local_proc: %llx, received a message from proc %llx, ACK", __FUNCTION__, local_proc.id, message.sender.id);
      break;
    }
    case TASK_STEAL_CONTINUE:
    {
      select_tasks_to_map_local = false;
      task_steal_request_t request = *(task_steal_request_t*)message.message;
      task_steal_request_queue.push_back(request);
      trigger_select_tasks_to_map(ctx);
      log_adapt_mapper.debug("%s, local_proc: %llx, received a message from proc %llx, CONTINUE", __FUNCTION__, local_proc.id, message.sender.id);
      break;
    }
    default: assert(false);
  }
}

//--------------------------------------------------------------------------
void AdaptiveMapper::select_task_options(const MapperContext ctx,
                                            const Task& task,
                                                  TaskOptions& output)
//--------------------------------------------------------------------------
{
#if defined USE_DEFAULT
  DefaultMapper::select_task_options(ctx, task, output);
#else 
  output.inline_task = false;
  output.stealable = true;
  output.map_locally = false;
  output.initial_proc = select_processor_by_id(local_proc.kind(), 0);
#endif
}


//--------------------------------------------------------------------------
void AdaptiveMapper::slice_task(const MapperContext      ctx,
                                   const Task&              task,
                                   const SliceTaskInput&    input,
                                         SliceTaskOutput&   output)
//--------------------------------------------------------------------------
{
#if defined USE_DEFAULT
  DefaultMapper::slice_task(ctx, task, input, output);
#else
  log_adapt_mapper.debug("%s, local_proc: %llx", __FUNCTION__, local_proc.id);
  // Iterate over all the points and send them all over the world
  output.slices.resize(input.domain.get_volume());
  unsigned idx = 0;
  switch (input.domain.get_dim())
  {
    case 1:
      {
        Rect<1> rect = input.domain;
        for (PointInRectIterator<1> pir(rect); pir(); pir++, idx++)
        {
          Rect<1> slice(*pir, *pir);
          Processor proc = select_processor_by_id(task.target_proc.kind(), 1);
          output.slices[idx] = TaskSlice(slice, proc,
              false/*recurse*/, true/*stealable*/);
          log_adapt_mapper.debug("index task: %p, original target proc: %llx, slice task process: %llx", &task, task.target_proc.id, proc.id);
        }
        break;
      }
    default:
      assert(false);
  }
#if 0
  Machine::ProcessorQuery all_procs(machine);
  all_procs.only_kind(local_proc.kind());
  std::vector<Processor> procs(all_procs.begin(), all_procs.end());
  switch (input.domain.get_dim())
  {
    case 1:
      {
        DomainT<1,coord_t> point_space = input.domain;
        Point<1,coord_t> num_points = 
                point_space.bounds.hi - point_space.bounds.lo;
        Point<1,coord_t> num_blocks((num_points.x+1) / num_tasks_per_slice);
       // assert(num_blocks.x == 4);
        DefaultMapper::default_decompose_points<1>(point_space, procs,
                                                   num_blocks, false/*recurse*/,
                                                   stealing_enabled, output.slices);
        break;
      }
    default:
      assert(false);
  }
#endif
#endif
}                                         

//--------------------------------------------------------------------------
void AdaptiveMapper::select_tasks_to_map(const MapperContext          ctx,
                                         const SelectMappingInput&    input,
                                               SelectMappingOutput&   output)
//--------------------------------------------------------------------------
{
  
  unsigned count = 0;
  
  if (select_tasks_to_map_local == true)
  {
    log_adapt_mapper.debug("%s, select_task, local_proc: %llx, ready_tasks size: %ld", __FUNCTION__, local_proc.id, input.ready_tasks.size());
    for (std::list<const Task*>::const_iterator it = 
          input.ready_tasks.begin(); (count < max_schedule_count) && 
          (it != input.ready_tasks.end()); it++)
    {
      const Task *task = *it; 
      if (RecursiveTaskArgument::is_task_recursiveable(task)) {
        // first, if there is task steal request, let's 
        
        if (recursive_tasks_scheduled >= max_recursive_tasks_to_schedule) {
          log_adapt_mapper.debug("%s, task find: %s, but not schedule, task_scheduled: %d", __FUNCTION__, task->get_task_name(), recursive_tasks_scheduled);

          continue;
        }
        // TODO: now, use a trick, slice task is sliced into 4 points, so with -ll:cpu 1, set to 4, -ll:cpu 2, set to 2.
        recursive_tasks_scheduled += num_tasks_per_slice;
        log_adapt_mapper.debug("%s, task find: %s, schedule, task_scheduled: %d, target proc: %llx", __FUNCTION__, task->get_task_name(), recursive_tasks_scheduled, task->target_proc.id);
      }
      
      output.map_tasks.insert(*it);
      //Processor target = select_processor_by_id(local_proc.kind(), 0);
      //output.relocate_tasks[*it] = target;
      count++;
    }
  } else {
    log_adapt_mapper.debug("%s, relocate_task, local_proc: %llx, ready_tasks size: %ld", __FUNCTION__, local_proc.id, input.ready_tasks.size());
    assert(task_steal_request_queue.size() > 0);
    task_steal_request_t &request = task_steal_request_queue.front();
  /*  for (std::list<const Task*>::const_iterator it = 
          input.ready_tasks.begin(); (count < max_schedule_count) && 
          (it != input.ready_tasks.end()); it++)
    {
      const Task *task = *it; 
      if (RecursiveTaskArgument::is_task_recursiveable(task)) {
        //Processor target = select_processor_by_id(local_proc.kind(), 0);
        output.relocate_tasks[*it] = request.target_proc;
        task_steal_request_queue.pop_front();
        break;
      }
    }*/
    
   // task_steal_request_t &request = task_steal_request_queue.front();
    std::list<const Task*>::const_iterator task_it = input.ready_tasks.begin();
    unsigned ready_tasks_size = input.ready_tasks.size();
    unsigned num_tasks_relocate = 0;
    while (task_steal_request_queue.size() > 0 && ready_tasks_size > 0) 
    {
      if (request.num_tasks <= ready_tasks_size) 
      {
        num_tasks_relocate = request.num_tasks;
      } 
      else
      {
        num_tasks_relocate = ready_tasks_size;
      }
      for (unsigned i = 0; i < num_tasks_relocate; i++) {
        assert(task_it != input.ready_tasks.end());
        output.relocate_tasks[*task_it] = request.target_proc;
        task_it ++;
      }
      ready_tasks_size -= num_tasks_relocate;
      task_steal_request_queue.pop_front();
      request = task_steal_request_queue.front();
    }
    
    select_tasks_to_map_local = true;
  }

  if (!defer_select_tasks_to_map.exists()) {
    defer_select_tasks_to_map = runtime->create_mapper_event(ctx);
  }
  output.deferral_event = defer_select_tasks_to_map;
}

//--------------------------------------------------------------------------
void AdaptiveMapper::map_task(const MapperContext         ctx,
                              const Task&                 task,
                              const MapTaskInput&         input,
                                    MapTaskOutput&        output)
//--------------------------------------------------------------------------
{
  // Pick a variant, then pick separate instances for all the 
  // fields in a region requirement
  const std::map<VariantID,Processor::Kind> &variant_kinds = 
    find_task_variants(ctx, task.task_id);
  std::vector<VariantID> variants;
  for (std::map<VariantID,Processor::Kind>::const_iterator it = 
        variant_kinds.begin(); it != variant_kinds.end(); it++)
  {
    if (task.target_proc.kind() == it->second)
      variants.push_back(it->first);
  }
  assert(!variants.empty());
  if (variants.size() > 1)
  {
    log_adapt_mapper.debug("%s, task: %s, task_target_proc: %llx, local_proc: %llx", __FUNCTION__, task.get_task_name(), task.target_proc.id, local_proc.id);
    bool task_recursiveable = RecursiveTaskArgument::is_task_recursiveable(&task);
    std::map<TaskID, bool>::iterator it = task_use_recursive.find(task.task_id);
    // the first time encounter this task
    if (it == task_use_recursive.end()) {
      task_use_recursive[task.task_id] = false;
    } 
 //   task_use_recursive[task.task_id] = false; 
    int chosen_v = 0;
    //printf("number of variants %d\n", variants.size());
    const int point = task.index_point.point_data[0];
    if (task_use_recursive[task.task_id] && task_recursiveable) {
      RecursiveTaskArgument::set_task_is_recursive_task(&task);
      chosen_v = 1;
    } else {
      chosen_v = 0;
    }
    output.chosen_variant = variants[chosen_v];
    
  }
  else
    output.chosen_variant = variants[0];
  
  // Let's ask for some profiling data to see the impact of our choices
  {
    using namespace ProfilingMeasurements;
    output.task_prof_requests.add_measurement<OperationStatus>();
    output.task_prof_requests.add_measurement<OperationTimeline>();
 //   output.task_prof_requests.add_measurement<RuntimeOverhead>();
  }
  default_map_task(ctx, task, input, output); 

  output.target_procs.clear();
  output.target_procs.push_back(local_proc);
}

//--------------------------------------------------------------------------
void AdaptiveMapper::select_steal_targets(const MapperContext         ctx,
                                          const SelectStealingInput&  input,
                                                SelectStealingOutput& output)
//--------------------------------------------------------------------------
{
 // Always send a steal request
  if (task_stealable_processor_list.size() == 0 || slow_down_mapper == true) {
    return;
  }
  Processor target = select_stealable_processor(local_proc.kind());
  //Processor target = select_processor_by_id(local_proc.kind(), 1);
  if (target != local_proc) {
   // output.targets.insert(target);
    task_steal_request_t request = {local_proc, 1};
    runtime->send_message(ctx, target, &request, sizeof(task_steal_request_t), TASK_STEAL_CONTINUE);
    log_adapt_mapper.debug("%s, local_proc: %llx, steal target %llx, stealable_list size %d", 
                           __FUNCTION__, local_proc.id, target.id,
                           task_stealable_processor_list.size());
    //assert(0);
  }
}

//--------------------------------------------------------------------------
void AdaptiveMapper::permit_steal_request(const MapperContext         ctx,
                                          const StealRequestInput&    input,
                                                StealRequestOutput&   output)
//--------------------------------------------------------------------------
{
  if (input.stealable_tasks.size() == 0) {
    return;
  }
  unsigned index = 1 % input.stealable_tasks.size();
  std::vector<const Task*>::const_iterator it = 
    input.stealable_tasks.begin();
  for (unsigned idx = 0; idx < index; idx++) it++;
  output.stolen_tasks.insert(*it);
  log_adapt_mapper.debug("%s, local_proc: %llx, stealable size %ld", __FUNCTION__, local_proc.id, input.stealable_tasks.size());
}

//--------------------------------------------------------------------------
void AdaptiveMapper::report_profiling(const MapperContext      ctx,
					                            const Task&              task,
					                            const TaskProfilingInfo& input)
//--------------------------------------------------------------------------
{
  // Local import of measurement names saves typing here without polluting
  // namespace for everybody else
  using namespace ProfilingMeasurements;

  // You are not guaranteed to get measurements you asked for, so make sure to
  // check the result of calls to get_measurement (or just call has_measurement
  // first).  Also, the call returns a copy of the result that you must delete
  // yourself.
  if (RecursiveTaskArgument::is_task_recursiveable(&task)) {
    OperationTimeline *timeline =
      input.profiling_responses.get_measurement<OperationTimeline>();
    if (timeline)
    {
      recursive_tasks_scheduled --;
    //  if (recursive_tasks_scheduled < 0) recursive_tasks_scheduled = 0;
      assert(recursive_tasks_scheduled >= 0);
      bool is_recursive_task = RecursiveTaskArgument::is_task_recursive_task(&task);
      // find the profiling history of task
      std::map<TaskID, task_profiling_t>::iterator it;
      it = task_profiling_history.find(task.task_id);
      if (it != task_profiling_history.end()) {
        task_profiling_t task_previous_profile = it->second;
        task_profiling_t task_profile;
        task_profile.is_recursive_task = is_recursive_task;
        task_profile.duration = timeline->end_time - timeline->start_time;
        task_profiling_history[task.task_id] = task_profile;
        if (task_profile.duration / task_previous_profile.duration > task_slowdown_allowance) {
         //  task_use_recursive[task.task_id] = true;
            if(slow_down_mapper == false) { 
              char *msg = "S"; 
              runtime->broadcast(ctx, msg, sizeof(char), TASK_STEAL_REQUEST);
              slow_down_mapper = true;
              printf("@@@@@@@@############$$$$$$$$$$$!!!!!!!!!!!!! i %llx become slow down\n", local_proc.id);
            }
        }
      } else {
        task_profiling_t task_profile;
        task_profile.is_recursive_task = is_recursive_task;
        task_profile.duration = timeline->end_time - timeline->start_time;
        task_profiling_history[task.task_id] = task_profile;
      }
      const Task *parent_task = task.parent_task;
      char *parent_task_name = "NULL";
      if (parent_task != NULL) {
        parent_task_name = (char*)parent_task->get_task_name();
      }

      log_adapt_mapper.debug("%s, task: %s: local_proc: %llx, ready=%lld start=%lld stop=%lld duration=%lld, parent: %s, is_recursive_task %d",
       __FUNCTION__,
  	   task.get_task_name(),
       local_proc.id,
  	   timeline->ready_time,
  	   timeline->start_time,
  	   timeline->end_time, timeline->end_time - timeline->start_time, parent_task_name, is_recursive_task);
      delete timeline;
      if (recursive_tasks_scheduled < max_recursive_tasks_to_schedule) {
        trigger_select_tasks_to_map(ctx);
      }
      
      Processor target = select_processor_by_id(local_proc.kind(), 1);
    //  if (target != local_proc) {
    //    char *msg = "A";
    //    runtime->send_message(ctx, target, msg, sizeof(char), TASK_STEAL_ACK);
    //  }
    }
    else {
      log_adapt_mapper.debug("No operation timeline for task %s", task.get_task_name());
    }
  }

}

//--------------------------------------------------------------------------
Processor AdaptiveMapper::select_processor_by_id(Processor::Kind kind, 
                                                 const unsigned int chosen_id)
//--------------------------------------------------------------------------
{
  Machine::ProcessorQuery mymachine(machine);
  mymachine.only_kind(kind);
  assert(chosen_id < mymachine.count());
 // printf("count %d\n", mymachine.count());
  Machine::ProcessorQuery::iterator it = mymachine.begin();
  for (unsigned int idx = 0; idx < chosen_id; idx++) it++;
  return (*it);
}

//--------------------------------------------------------------------------
Processor AdaptiveMapper::select_stealable_processor(Processor::Kind kind)
//--------------------------------------------------------------------------
{
  assert(task_stealable_processor_list.size() > 0);
  std::set<Processor>::iterator it = task_stealable_processor_list.begin();
  int chosen_id = default_generate_random_integer() % task_stealable_processor_list.size();
  int idx = 0;
  while(it != task_stealable_processor_list.end() && idx < chosen_id) 
  {
    if ((*it).kind() == kind) 
    {
      idx ++;
    }
    it ++;
  }
  assert(it != task_stealable_processor_list.end());
  return (*it);
}

//--------------------------------------------------------------------------
const std::map<VariantID,Processor::Kind>& AdaptiveMapper::find_task_variants(
                                          MapperContext ctx, TaskID task_id)
//--------------------------------------------------------------------------
{
  std::map<TaskID,std::map<VariantID,Processor::Kind> >::const_iterator
    finder = variant_processor_kinds.find(task_id);
  if (finder != variant_processor_kinds.end())
    return finder->second;
  std::vector<VariantID> valid_variants;
  runtime->find_valid_variants(ctx, task_id, valid_variants);
  std::map<VariantID,Processor::Kind> kinds;
  for (std::vector<VariantID>::const_iterator it = valid_variants.begin();
        it != valid_variants.end(); it++)
  {
    const ExecutionConstraintSet &constraints = 
      runtime->find_execution_constraints(ctx, task_id, *it);
    if (constraints.processor_constraint.is_valid())
      kinds[*it] = constraints.processor_constraint.get_kind();
    else
      kinds[*it] = Processor::LOC_PROC; // assume CPU
  }
  std::map<VariantID,Processor::Kind> &result = 
    variant_processor_kinds[task_id];
  result = kinds;
  return result;
}

//--------------------------------------------------------------------------
void AdaptiveMapper::default_map_task(const MapperContext         ctx,
                                         const Task&                 task,
                                         const MapTaskInput&         input,
                                               MapTaskOutput&        output)
//--------------------------------------------------------------------------
{
  output.task_priority = default_policy_select_task_priority(ctx, task);
  output.postmap_task = false;
  // Figure out our target processors
  default_policy_select_target_processors(ctx, task, output.target_procs);

  // See if we have an inner variant, if we do virtually map all the regions
  // We don't even both caching these since they are so simple
#if 0  
  if (chosen.is_inner)
  {
    // Check to see if we have any relaxed coherence modes in which
    // case we can no longer do virtual mappings so we'll fall through
    bool has_relaxed_coherence = false;
    for (unsigned idx = 0; idx < task.regions.size(); idx++)
    {
      if (task.regions[idx].prop != EXCLUSIVE)
      {
        has_relaxed_coherence = true;
        break;
      }
    }
    if (!has_relaxed_coherence)
    {
      std::vector<unsigned> reduction_indexes;
      for (unsigned idx = 0; idx < task.regions.size(); idx++)
      {
        // As long as this isn't a reduction-only region requirement
        // we will do a virtual mapping, for reduction-only instances
        // we will actually make a physical instance because the runtime
        // doesn't allow virtual mappings for reduction-only privileges
        if (task.regions[idx].privilege == REDUCE)
          reduction_indexes.push_back(idx);
        else
          output.chosen_instances[idx].push_back(
              PhysicalInstance::get_virtual_instance());
      }
      if (!reduction_indexes.empty())
      {
        const TaskLayoutConstraintSet &layout_constraints =
            runtime->find_task_layout_constraints(ctx,
                                  task.task_id, output.chosen_variant);
        for (std::vector<unsigned>::const_iterator it = 
              reduction_indexes.begin(); it != 
              reduction_indexes.end(); it++)
        {
          Memory target_memory = default_policy_select_target_memory(ctx,
                                                     task.target_proc,
                                                     task.regions[*it]);
          std::set<FieldID> copy = task.regions[*it].privilege_fields;
          if (!default_create_custom_instances(ctx, task.target_proc,
              target_memory, task.regions[*it], *it, copy, 
              layout_constraints, false/*needs constraint check*/, 
              output.chosen_instances[*it]))
          {
            default_report_failed_instance_creation(task, *it, 
                                        task.target_proc, target_memory);
          }
        }
      }
      return;
    }
  }
#endif
  // Should we cache this task?
  CachedMappingPolicy cache_policy =
    default_policy_select_task_cache_policy(ctx, task);

  // First, let's see if we've cached a result of this task mapping
  const unsigned long long task_hash = compute_task_hash(task);
  std::pair<TaskID,Processor> cache_key(task.task_id, task.target_proc);
  std::map<std::pair<TaskID,Processor>,
           std::list<CachedTaskMapping> >::const_iterator 
    finder = cached_task_mappings.find(cache_key);
  // This flag says whether we need to recheck the field constraints,
  // possibly because a new field was allocated in a region, so our old
  // cached physical instance(s) is(are) no longer valid
  bool needs_field_constraint_check = false;
  if (cache_policy == DEFAULT_CACHE_POLICY_ENABLE && finder != cached_task_mappings.end())
  {
    bool found = false;
    bool has_reductions = false;
    // Iterate through and see if we can find one with our variant and hash
    for (std::list<CachedTaskMapping>::const_iterator it = 
          finder->second.begin(); it != finder->second.end(); it++)
    {
      if ((it->variant == output.chosen_variant) &&
          (it->task_hash == task_hash))
      {
        // Have to copy it before we do the external call which 
        // might invalidate our iterator
        output.chosen_instances = it->mapping;
        has_reductions = it->has_reductions;
        found = true;
        break;
      }
    }
    if (found)
    {
      // If we have reductions, make those instances now since we
      // never cache the reduction instances
      if (has_reductions)
      {
        const TaskLayoutConstraintSet &layout_constraints =
          runtime->find_task_layout_constraints(ctx,
                              task.task_id, output.chosen_variant);
        for (unsigned idx = 0; idx < task.regions.size(); idx++)
        {
          if (task.regions[idx].privilege == REDUCE)
          {
            Memory target_memory = default_policy_select_target_memory(ctx,
                                                     task.target_proc,
                                                     task.regions[idx]);
            std::set<FieldID> copy = task.regions[idx].privilege_fields;
            if (!default_create_custom_instances(ctx, task.target_proc,
                target_memory, task.regions[idx], idx, copy, 
                layout_constraints, needs_field_constraint_check, 
                output.chosen_instances[idx]))
            {
              default_report_failed_instance_creation(task, idx, 
                                          task.target_proc, target_memory);
            }
          }
        }
      }
      // See if we can acquire these instances still
      if (runtime->acquire_and_filter_instances(ctx, 
                                                 output.chosen_instances))
        return;
      // We need to check the constraints here because we had a
      // prior mapping and it failed, which may be the result
      // of a change in the allocated fields of a field space
      needs_field_constraint_check = true;
      // If some of them were deleted, go back and remove this entry
      // Have to renew our iterators since they might have been
      // invalidated during the 'acquire_and_filter_instances' call
      default_remove_cached_task(ctx, output.chosen_variant,
                    task_hash, cache_key, output.chosen_instances);
    }
  }
  // We didn't find a cached version of the mapping so we need to 
  // do a full mapping, we already know what variant we want to use
  // so let's use one of the acceleration functions to figure out
  // which instances still need to be mapped.
  std::vector<std::set<FieldID> > missing_fields(task.regions.size());
  runtime->filter_instances(ctx, task, output.chosen_variant,
                             output.chosen_instances, missing_fields);
  // Track which regions have already been mapped 
  std::vector<bool> done_regions(task.regions.size(), false);
  if (!input.premapped_regions.empty())
    for (std::vector<unsigned>::const_iterator it = 
          input.premapped_regions.begin(); it != 
          input.premapped_regions.end(); it++)
      done_regions[*it] = true;
  const TaskLayoutConstraintSet &layout_constraints = 
    runtime->find_task_layout_constraints(ctx, 
                          task.task_id, output.chosen_variant);
  // Now we need to go through and make instances for any of our
  // regions which do not have space for certain fields
  bool has_reductions = false;
  for (unsigned idx = 0; idx < task.regions.size(); idx++)
  {
    if (done_regions[idx])
      continue;
    // Skip any empty regions
    if ((task.regions[idx].privilege == NO_ACCESS) ||
        (task.regions[idx].privilege_fields.empty()) ||
        missing_fields[idx].empty())
      continue;
    // See if this is a reduction      
    Memory target_memory = default_policy_select_target_memory(ctx,
                                                     task.target_proc,
                                                     task.regions[idx]);
    if (task.regions[idx].privilege == REDUCE)
    {
      has_reductions = true;
      if (!default_create_custom_instances(ctx, task.target_proc,
              target_memory, task.regions[idx], idx, missing_fields[idx],
              layout_constraints, needs_field_constraint_check,
              output.chosen_instances[idx]))
      {
        default_report_failed_instance_creation(task, idx, 
                                    task.target_proc, target_memory);
      }
      continue;
    }
    // Did the application request a virtual mapping for this requirement?
    if ((task.regions[idx].tag & DefaultMapper::VIRTUAL_MAP) != 0)
    {
      PhysicalInstance virt_inst = PhysicalInstance::get_virtual_instance();
      output.chosen_instances[idx].push_back(virt_inst);
      continue;
    }
    // Check to see if any of the valid instances satisfy this requirement
    {
      std::vector<PhysicalInstance> valid_instances;

      for (std::vector<PhysicalInstance>::const_iterator
             it = input.valid_instances[idx].begin(),
             ie = input.valid_instances[idx].end(); it != ie; ++it)
      {
        if (it->get_location() == target_memory)
          valid_instances.push_back(*it);
      }

      std::set<FieldID> valid_missing_fields;
      runtime->filter_instances(ctx, task, idx, output.chosen_variant,
                                valid_instances, valid_missing_fields);

#ifndef NDEBUG
      bool check =
#endif
        runtime->acquire_and_filter_instances(ctx, valid_instances);
      assert(check);

      output.chosen_instances[idx] = valid_instances;
      missing_fields[idx] = valid_missing_fields;

      if (missing_fields[idx].empty())
        continue;
    }
    // Otherwise make normal instances for the given region
    if (!default_create_custom_instances(ctx, task.target_proc,
            target_memory, task.regions[idx], idx, missing_fields[idx],
            layout_constraints, needs_field_constraint_check,
            output.chosen_instances[idx]))
    {
      default_report_failed_instance_creation(task, idx,
                                  task.target_proc, target_memory);
    }
  }
  if (cache_policy == DEFAULT_CACHE_POLICY_ENABLE) {
    // Now that we are done, let's cache the result so we can use it later
    std::list<CachedTaskMapping> &map_list = cached_task_mappings[cache_key];
    map_list.push_back(CachedTaskMapping());
    CachedTaskMapping &cached_result = map_list.back();
    cached_result.task_hash = task_hash;
    cached_result.variant = output.chosen_variant;
    cached_result.mapping = output.chosen_instances;
    cached_result.has_reductions = has_reductions;
    // We don't ever save reduction instances in our cache
    if (has_reductions) {
      for (unsigned idx = 0; idx < task.regions.size(); idx++) {
        if (task.regions[idx].privilege != REDUCE)
          continue;
        cached_result.mapping[idx].clear();
      }
    }
  }  
}

//--------------------------------------------------------------------------
void AdaptiveMapper::trigger_select_tasks_to_map(const MapperContext ctx)
//--------------------------------------------------------------------------
{
  if (defer_select_tasks_to_map.exists()){
    log_adapt_mapper.debug("%s, local_proc: %llx", __FUNCTION__, local_proc.id);
    MapperEvent temp_event = defer_select_tasks_to_map;
    defer_select_tasks_to_map = MapperEvent();
    runtime->trigger_mapper_event(ctx, temp_event);
  } else {
//    log_adapt_mapper.debug("proc %llx: try to trigger but event not exist",
//                           local_proc.id);
  }
}


/*
 * Everything below here is the standard daxpy example
 * except for the registration of the callback function
 * for creating custom mappers which is explicitly commented
 * and the call to select_tunable_value to determine the number
 * of sub-regions.
 */
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 1024; 
  int num_subregions = 4;
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-b"))
        num_subregions = atoi(command_args.argv[++i]);
    }
  }

  printf("Running daxpy for %d elements...\n", num_elements);
  printf("Partitioning data into %d sub-regions...\n", num_subregions);

  Rect<1> elem_rect(0,num_elements-1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace input_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(double),FID_X);
    allocator.allocate_field(sizeof(double),FID_Y);
  }
  FieldSpace output_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, output_fs);
    allocator.allocate_field(sizeof(double),FID_Z);
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is, output_fs);

  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);

  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);

  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, ip);

  ArgumentMap arg_map;

  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, 
                              TaskArgument(NULL, 0), arg_map);
  init_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/, 
                        WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.add_field(0, FID_X);
  runtime->execute_index_space(ctx, init_launcher);

  init_launcher.region_requirements[0].privilege_fields.clear();
  init_launcher.region_requirements[0].instance_fields.clear();
  init_launcher.add_field(0, FID_Y);
  runtime->execute_index_space(ctx, init_launcher);

  const double alpha = drand48();
  RecursiveTaskArgument daxpy_task_args(&alpha, sizeof(double), 1, 0);
  
  double t_start = get_cur_time();
  for (int ct = 0; ct < 5; ct++) {
  
  IndexLauncher daxpy_launcher(DAXPY_TASK_ID, color_is,
                TaskArgument(daxpy_task_args.get_args(), daxpy_task_args.get_arglen()), arg_map);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, input_lr));
  daxpy_launcher.add_field(0, FID_X);
  daxpy_launcher.add_field(0, FID_Y);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, output_lr));
  daxpy_launcher.add_field(1, FID_Z);
  //daxpy_launcher.set_recursiveable(true);
  FutureMap fm = runtime->execute_index_space(ctx, daxpy_launcher);
  //fm.wait_all_results();
  double t_end = get_cur_time();
  double sim_time = (t_end - t_start);
  printf("ELAPSED TIME = %7.3f s\n", sim_time);
}
  
  RecursiveTaskArgument check_task_args(&alpha, sizeof(double), 0, 0);
                    
  TaskLauncher check_launcher(CHECK_TASK_ID, TaskArgument(check_task_args.get_args(), check_task_args.get_arglen()));
  check_launcher.add_region_requirement(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  check_launcher.region_requirements[0].add_field(FID_X);
  check_launcher.region_requirements[0].add_field(FID_Y);
  check_launcher.add_region_requirement(
      RegionRequirement(output_lr, READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.region_requirements[1].add_field(FID_Z);
  runtime->execute_task(ctx, check_launcher);

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_field_space(ctx, output_fs);
  runtime->destroy_index_space(ctx, is);
}

void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];
  printf("Initializing field %d for block %d...\n", fid, point);

  const FieldAccessor<WRITE_DISCARD,double,1> acc(regions[0], fid);
  // Note here that we get the domain for the subregion for
  // this task from the runtime which makes it safe for running
  // both as a single task and as part of an index space of tasks.
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc[*pir] = drand48();
}

void daxpy_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  size_t user_arglen = RecursiveTaskArgument::get_usr_arglen(task->arglen);
  assert(user_arglen == sizeof(double));
  const double alpha = *((const double*)RecursiveTaskArgument::get_usr_args(task->args));
  const int point = task->index_point.point_data[0];

  const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);
  const FieldAccessor<READ_ONLY,double,1> acc_y(regions[0], FID_Y);
  const FieldAccessor<WRITE_DISCARD,double,1> acc_z(regions[1], FID_Z);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
  task->regions[0].region.get_index_space());
  int size = rect.hi.x - rect.lo.x + 1;
  
  int is_recursive_task = 0;
  if (RecursiveTaskArgument::is_task_recursive_task(task)) {
    is_recursive_task = 1;
  }
  
  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);
  pthread_t         self;
  self = pthread_self();

  printf("Running daxpy computation with alpha %.8g for point %d size %d, is_recursive_task %d, host %s, thread %lld, current_proc %llx, target_proc %llx\n", 
          alpha, point, size, is_recursive_task, hostname, self, task->current_proc.id, task->target_proc.id);

  if (point == 1 && is_recursive_task == 0) sleep(5);
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc_z[*pir] = alpha * acc_x[*pir] + acc_y[*pir];
  
}

void daxpy_task_split(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  size_t user_arglen = RecursiveTaskArgument::get_usr_arglen(task->arglen);
  assert(user_arglen == sizeof(double));
  const double alpha = *((const double*)RecursiveTaskArgument::get_usr_args(task->args));
  const int point = task->index_point.point_data[0];

  Rect<1> rect = runtime->get_index_space_domain(ctx,
  task->regions[0].region.get_index_space());
  int size = rect.hi.x - rect.lo.x + 1;

  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);
  
  printf("Running daxpy split with alpha %.8g for point %d size %d, hostname %s...\n", 
          alpha, point, size, hostname);
  Rect<1> color_bounds(0, 3);
  IndexSpace is = task->regions[0].region.get_index_space();
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);
  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  runtime->attach_name(ip, "ip");

  LogicalRegion input_lr = regions[0].get_logical_region();
  LogicalRegion output_lr = regions[1].get_logical_region();
  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  runtime->attach_name(input_lp, "input_lp");
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, ip);
  runtime->attach_name(output_lp, "output_lp");

  ArgumentMap arg_map;
  RecursiveTaskArgument task_args(&alpha, sizeof(double), 0, 1);
  
  IndexLauncher daxpy_launcher(DAXPY_TASK_ID, color_is,
                TaskArgument(task_args.get_args(), task_args.get_arglen()), arg_map);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, input_lr));
  daxpy_launcher.region_requirements[0].add_field(FID_X);
  daxpy_launcher.region_requirements[0].add_field(FID_Y);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, output_lr));
  daxpy_launcher.region_requirements[1].add_field(FID_Z);
  //daxpy_launcher.set_is_recursive_task(true);
  runtime->execute_index_space(ctx, daxpy_launcher);
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  size_t user_arglen = RecursiveTaskArgument::get_usr_arglen(task->arglen);
  assert(user_arglen == sizeof(double));
  const double alpha = *((const double*)RecursiveTaskArgument::get_usr_args(task->args));

  const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);
  const FieldAccessor<READ_ONLY,double,1> acc_y(regions[0], FID_Y);
  const FieldAccessor<READ_ONLY,double,1> acc_z(regions[1], FID_Z);

  printf("Checking results...");
  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  bool all_passed = true;
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
  {
    double expected = alpha * acc_x[*pir] + acc_y[*pir];
    double received = acc_z[*pir];
    // Probably shouldn't check for floating point equivalence but
    // the order of operations are the same should they should
    // be bitwise equal.
    if (expected != received)
      all_passed = false;
  }
  if (all_passed)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");
}

int main(int argc, char **argv)
{
  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);
  printf("My process ID : %d, host %s\n", getpid(), hostname);
  //sleep(10);
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INIT_FIELD_TASK_ID, "init_field");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_field_task>(registrar, "init_field");
  }

  {
    TaskVariantRegistrar registrar(DAXPY_TASK_ID, "daxpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<daxpy_task>(registrar, "daxpy");
  }
  
  {
    TaskVariantRegistrar registrar(DAXPY_TASK_ID, "daxpy");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<daxpy_task_split>(registrar, "daxpy");
  }

  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<check_task>(registrar, "check");
  }

  // Here is where we register the callback function for 
  // creating custom mappers.
  Runtime::add_registration_callback(mapper_registration);
  
  printf("done with reg\n");

  return Runtime::start(argc, argv);
}
