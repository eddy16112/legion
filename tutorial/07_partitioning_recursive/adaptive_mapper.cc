#include "adaptive_mapper.h"

int is_slow_down_proc(unsigned long long proc_id)
{
	unsigned long long slow_down_proc_id_1 = 0x1d00010000000001;
	unsigned long long slow_down_proc_id_2 = 0x1d00010000000001;
	if (proc_id == slow_down_proc_id_1 || proc_id == slow_down_proc_id_2) {
		return 1;
	} else {
		return 0;
	}
}

static LegionRuntime::Logger::Category log_adapt_mapper("adapt_mapper");

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
	return true;
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

RecursiveTaskMapperShared::RecursiveTaskMapperShared()
{
  task_slowdown_allowance = 2;
  max_recursive_tasks_to_schedule = 1;
  recursive_tasks_scheduled = 0;
}


/* mapper */
void mapper_registration(Machine machine, Runtime *rt,
                          const std::set<Processor> &local_procs)
{
  RecursiveTaskMapperShared *mapper_shared = new RecursiveTaskMapperShared();
  printf("local procs size %d\n", local_procs.size());
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    long long proc_id = (*it).id;
    unsigned node_id = proc_id >> 40;
    unsigned pid = proc_id & 0xffffffffff;
    printf("proc id %llx, node %x, proc %x\n", proc_id, node_id, pid);
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
  max_recursive_tasks_to_schedule = 2;
  recursive_tasks_scheduled = 0;
	select_tasks_to_map_relocate = false;
  num_tasks_per_slice = 1;
  task_stealable_processor_list.clear();
  slow_down_mapper = false;
  
  num_ready_tasks = 0;
  min_ready_tasks_to_enable_steal = 1;
  
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
	
  local_node_id = (local_proc.id) >> 40;	
  
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
				log_adapt_mapper.debug("%s, local_proc: %llx, received a message from proc %llx, num_ready_tasks %d, STEAL", __FUNCTION__, local_proc.id, message.sender.id, num_ready_tasks);
        // add sender to stealable processor list
        std::set<Processor>::iterator it;
        it = task_stealable_processor_list.find(message.sender);
        if (it == task_stealable_processor_list.end()) {
          task_stealable_processor_list.insert(message.sender);
        }
        if (num_ready_tasks <= min_ready_tasks_to_enable_steal && slow_down_mapper == false) {
          task_steal_request_t request = {local_proc, 2};
          runtime->send_message(ctx, message.sender, &request, sizeof(task_steal_request_t), TASK_STEAL_ACK);
          log_adapt_mapper.debug("%s, local_proc: %llx, send a message to proc %llx, num_ready_tasks %d, STEAL_ACK", __FUNCTION__, local_proc.id, message.sender.id, num_ready_tasks);
        }
      }
      break;
    }
    case TASK_STEAL_ACK:
    {
      select_tasks_to_map_relocate = true;
      task_steal_request_t request = *(task_steal_request_t*)message.message;
      task_steal_request_queue.push_back(request);
      trigger_select_tasks_to_map(ctx);
      log_adapt_mapper.debug("%s, local_proc: %llx, received a message from proc %llx, ACK", __FUNCTION__, local_proc.id, message.sender.id);
      break;
    }
    case TASK_STEAL_CONTINUE:
    {
			select_tasks_to_map_relocate = true;
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
  printf("select task options %p, local_proc: %llx\n", &task, local_proc.id);
#if defined USE_DEFAULT
  DefaultMapper::select_task_options(ctx, task, output);
#else 
  output.inline_task = false;
  output.stealable = true;
  output.map_locally = false;
  output.initial_proc = select_processor_by_id(local_proc.kind(), 0);
  
  //DefaultMapper::select_task_options(ctx, task, output);
  //output.stealable = true;
#endif
}


//--------------------------------------------------------------------------
void AdaptiveMapper::slice_task(const MapperContext      ctx,
                                const Task&              task,
                                const SliceTaskInput&    input,
                                      SliceTaskOutput&   output)
//--------------------------------------------------------------------------
{
  printf("slice_task %p, local_proc: %llx\n", &task, local_proc.id);
#if defined USE_DEFAULT
  DefaultMapper::slice_task(ctx, task, input, output);
#else
#if 0
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
#endif
  //#if 0
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
  //#endif
#endif
}                                         

//--------------------------------------------------------------------------
void AdaptiveMapper::select_tasks_to_map(const MapperContext          ctx,
                                         const SelectMappingInput&    input,
                                               SelectMappingOutput&   output)
//--------------------------------------------------------------------------
{
  
  unsigned count = 0;
	
	std::list<const Task*>::const_iterator task_it = input.ready_tasks.begin();
	unsigned ready_tasks_size = input.ready_tasks.size();
  
	if (select_tasks_to_map_relocate == true)
	{
    log_adapt_mapper.debug("%s, relocate_task, local_proc: %llx, ready_tasks size: %ld", __FUNCTION__, local_proc.id, input.ready_tasks.size());
   // printf("relocate_task, local_proc: %llx, ready_tasks size: %ld\n", local_proc.id, input.ready_tasks.size());
    assert(task_steal_request_queue.size() > 0);
    task_steal_request_t &request = task_steal_request_queue.front();
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
    
    select_tasks_to_map_relocate = false;	
	}
  {
    log_adapt_mapper.debug("%s, select_task, local_proc: %llx, ready_tasks size: %ld", __FUNCTION__, local_proc.id, input.ready_tasks.size());
    while((recursive_tasks_scheduled < max_recursive_tasks_to_schedule) && 
          (ready_tasks_size > 0))
    {
      const Task *task = *task_it; 
			/*
      if (RecursiveTaskArgument::is_task_recursiveable(task)) {
        if (recursive_tasks_scheduled >= max_recursive_tasks_to_schedule) {
          log_adapt_mapper.debug("%s, task find: %s, but not schedule, task_scheduled: %d", __FUNCTION__, task->get_task_name(), recursive_tasks_scheduled);

          continue;
        }
        // TODO: now, use a trick, slice task is sliced into 4 points, so with -ll:cpu 1, set to 4, -ll:cpu 2, set to 2.
        recursive_tasks_scheduled += num_tasks_per_slice;
        log_adapt_mapper.debug("%s, task find: %s, schedule, task_scheduled: %d, target proc: %llx", __FUNCTION__, task->get_task_name(), recursive_tasks_scheduled, task->target_proc.id);
      }*/
			if (task->is_index_space) {
				recursive_tasks_scheduled += num_tasks_per_slice;
			} else {
				recursive_tasks_scheduled += 1;
			}
      
      output.map_tasks.insert(*task_it);
			task_it ++;
			ready_tasks_size --;
      count++;
    }
    num_ready_tasks = input.ready_tasks.size() - count;
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
	DefaultMapper::map_task(ctx, task, input, output);
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
  if (task_stealable_processor_list.size() == 0 || slow_down_mapper == true || num_ready_tasks > min_ready_tasks_to_enable_steal) {
    if (slow_down_mapper == false) printf("local %llx can not steal num_ready %d, min %d\n", local_proc.id, num_ready_tasks, min_ready_tasks_to_enable_steal);
		return;
  }
  Processor target = select_stealable_processor(local_proc.kind());
  //Processor target = select_processor_by_id(local_proc.kind(), 1);
  if (target != local_proc) {
		printf("local %llx CAN steal num_ready %d, min %d\n", local_proc.id, num_ready_tasks, min_ready_tasks_to_enable_steal);
   // output.targets.insert(target);
    task_steal_request_t request = {local_proc, 1};
    runtime->send_message(ctx, target, &request, sizeof(task_steal_request_t), TASK_STEAL_CONTINUE);
    log_adapt_mapper.debug("%s, local_proc: %llx, steal target %llx, stealable_list size %d, ready_tasks %d", 
                           __FUNCTION__, local_proc.id, target.id,
                           task_stealable_processor_list.size(), num_ready_tasks);
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
	recursive_tasks_scheduled --;
	assert(recursive_tasks_scheduled >= 0);
  if (RecursiveTaskArgument::is_task_recursiveable(&task)) {
    OperationTimeline *timeline =
      input.profiling_responses.get_measurement<OperationTimeline>();
    if (timeline)
    {
      
    //  if (recursive_tasks_scheduled < 0) recursive_tasks_scheduled = 0;
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
        //if (task_profile.duration / task_previous_profile.duration > task_slowdown_allowance) {
				if (is_slow_down_proc(local_proc.id)) {
       //     task_use_recursive[task.task_id] = true;
            if(slow_down_mapper == false) { 
              char *msg = "S"; 
				  //    runtime->broadcast(ctx, msg, sizeof(char), TASK_STEAL_REQUEST);
						  std::set<Processor> all_procs;
						  machine.get_all_processors(all_procs);
						  for (std::set<Processor>::const_iterator it = all_procs.begin();
						        it != all_procs.end(); it++)
						  {
								Processor target_proc = *it;
								if (is_on_same_node_not_util(target_proc) && target_proc.id != local_proc.id) {
									printf("target_proc %llx, local_proc %llx\n", target_proc.id, local_proc.id);
									runtime->send_message(ctx, target_proc, msg, sizeof(char), TASK_STEAL_REQUEST);
								}
							}
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

      log_adapt_mapper.debug("%s, task: %s: local_proc: %llx, local_node %x, ready=%lld start=%lld stop=%lld duration=%lld, parent: %s, is_recursive_task %d",
       __FUNCTION__,
  	   task.get_task_name(),
       local_proc.id,
			 local_node_id,
  	   timeline->ready_time,
  	   timeline->start_time,
  	   timeline->end_time, timeline->end_time - timeline->start_time, parent_task_name, is_recursive_task);
      delete timeline;
      
     // Processor target = select_processor_by_id(local_proc.kind(), 1);
    //  if (target != local_proc) {
    //    char *msg = "A";
    //    runtime->send_message(ctx, target, msg, sizeof(char), TASK_STEAL_ACK);
    //  }
    }
    else {
      log_adapt_mapper.debug("No operation timeline for task %s", task.get_task_name());
    }
  }
  if (recursive_tasks_scheduled < max_recursive_tasks_to_schedule) {
    trigger_select_tasks_to_map(ctx);
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

//--------------------------------------------------------------------------
bool AdaptiveMapper::is_on_same_node_not_util(Processor &proc)
//--------------------------------------------------------------------------
{
	if (proc.kind() == Processor::UTIL_PROC) {
		return false;
	}
  unsigned node_id = (proc.id) >> 40;
	if (node_id == local_node_id) {
		return true;
	}	else {
		return false;
	}
}
