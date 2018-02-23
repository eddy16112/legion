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

#include "adaptive_mapper.h"

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



double get_cur_time() {
  struct timeval   tv;
  struct timezone  tz;
  double cur_time;
  
  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;
  
  return cur_time;
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

  printf("Checking results... %llx, ", task->target_proc.id);
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
  sleep(10);
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
