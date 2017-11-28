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
#include "legion.h"

#include "test_mapper.h"
#include "default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

/*
 * In this example, we perform the same
 * daxpy computation as example 07.  While
 * the source code for the actual computation
 * between the two examples is identical, we 
 * show to create a custom mapper that changes 
 * the mapping of the computation onto the hardware.
 */

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

/*
 * One of the primary goals of Legion is
 * to make it easy to remap applications
 * onto different hardware.  Up to this point
 * all of our applications have been mapped
 * by the DefaultMapper that we provide.  The
 * DefaultMapper class provides heuristics for
 * performing mappings that are good but often
 * not optimal for a specific application or
 * architecture.   By creating a custom mapper
 * programmers can make application- or
 * architecture-specific mapping decisions.
 * Furthermore, many custom mappers can be
 * used to map the same application onto many
 * different architectures without having to
 * modify the primary application code.
 *
 * A common concern when mapping applications
 * onto a target machine is whether or not
 * mapping impacts correctness.  In Legion
 * all mapping decisions are orthogonal to
 * correctness.  In cases when correctness may
 * be impacted by a mapping decision (e.g. a
 * mapper maps a physical region for a task
 * into a memory not visible from the target
 * processor), then the Legion runtime will
 * notify the mapper that it tried to perform
 * an illegal mapping and allow it to retry.
 *
 * To introduce how to write a custom mapper
 * we'll implement an adversarial mapper 
 * that makes random mapping decisions
 * designed to stress the Legion runtime. 
 * We'll report the chosen mapping decisions
 * to show that Legion computes the correct
 * answer regardless of the mapping.
 */

// Mappers are classes that implement the
// mapping interface declared in legion.h.
// Legion provides a default implementation
// of this interface defined by the
// DefaultMapper class.  Programmers can
// implement custom mappers either by 
// extending the DefaultMapper class
// or by declaring their own class which
// implements the mapping interface from
// scratch.  Here we will extend the
// DefaultMapper which means that we only
// need to override the methods that we
// want to in order to change our mapping.
// In this example, we'll override four
// of the mapping calls to illustrate
// how they work.
class AdversarialMapper : public DefaultMapper {
public:
  AdversarialMapper(Machine machine, 
      Runtime *rt, Processor local);
public:
  const std::map<VariantID,Processor::Kind>& find_task_variants(
                                            MapperContext ctx, TaskID task_id);
  
  void map_constrained_requirement(MapperContext ctx,
      const RegionRequirement &req, MappingKind mapping_kind, 
      const std::vector<LayoutConstraintID> &constraints,
      std::vector<PhysicalInstance> &chosen_instances, Processor restricted);
      
  void map_random_requirement(MapperContext ctx,
      const RegionRequirement &req, 
      std::vector<PhysicalInstance> &chosen_instances, Processor restricted);

  virtual void map_task(const MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                              MapTaskOutput& output);
  virtual void report_profiling(const MapperContext      ctx,
				const Task&              task,
				const TaskProfilingInfo& input);
protected:
      long generate_random_integer(void) const 
        { return default_generate_random_integer(); }

protected:
  std::map<TaskID,std::map<VariantID,
                            Processor::Kind> > variant_processor_kinds;
};
/*
class PartitioningMapper : public DefaultMapper {
public:
  PartitioningMapper(Machine machine,
      Runtime *rt, Processor local);
public:
  virtual void select_tunable_value(const MapperContext ctx,
                                    const Task& task,
                                    const SelectTunableInput& input,
                                          SelectTunableOutput& output);
};
*/

// Mappers are created after the Legion runtime
// starts but before the application begins 
// executing.  To create mappers the application
// registers a callback function for the runtime
// to perform prior to beginning execution.  In
// this example we call it the 'mapper_registration'
// function.  (See below for where we register
// this callback with the runtime.)  The callback
// function must have this type, which allows the
// runtime to pass the necessary parameters in
// for creating new mappers.
//
// In Legion, mappers are identified by a MapperID.
// Zero is reserved for the DefaultMapper, but 
// other mappers can replace it by using the 
// 'replace_default_mapper' call.  Other mappers
// can be registered with their own IDs using the
// 'add_mapper' method call on the runtime.
//
// The model for Legion is that there should always
// be one mapper for every processor in the system.
// This guarantees that there is never contention
// for the mappers because multiple processors need
// to access the same mapper object.  When the
// runtime invokes the 'mapper_registration' callback,
// it gives a list of local processors which 
// require a mapper if a new mapper class is to be
// created.  In a multi-node setting, the runtime
// passes in a subset of the processors for which
// mappers need to be created on the local node.
//
// Here we override the DefaultMapper ID so that
// all tasks that normally would have used the
// DefaultMapper will now use our AdversarialMapper.
// We create one new mapper for each processor
// and register it with the runtime.
void mapper_registration(Machine machine, Runtime *rt,
                          const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(
        new AdversarialMapper(machine, rt, *it), *it);
  }
}

// Here is the constructor for our adversarial mapper.
// We'll use the constructor to illustrate how mappers can
// get access to information regarding the current machine.
AdversarialMapper::AdversarialMapper(Machine m, 
                                     Runtime *rt, Processor p)
  : DefaultMapper(rt->get_mapper_runtime(), m, p) // pass arguments through to TestMapper 
{
  // The machine object is a singleton object that can be
  // used to get information about the underlying hardware.
  // The machine pointer will always be provided to all
  // mappers, but can be accessed anywhere by the static
  // member method Machine::get_machine().  Here we get
  // a reference to the set of all processors in the machine
  // from the machine object.  Note that the Machine object
  // actually comes from the Legion low-level runtime, most
  // of which is not directly needed by application code.
  // Typedefs in legion_types.h ensure that all necessary
  // types for querying the machine object are in scope
  // in the Legion namespace.
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

    // The Legion machine model represented by the machine object
    // can be thought of as a graph with processors and memories
    // as the two kinds of nodes.  There are two kinds of edges
    // in this graph: processor-memory edges and memory-memory
    // edges.  An edge between a processor and a memory indicates
    // that the processor can directly perform load and store
    // operations to that memory.  Memory-memory edges indicate
    // that data movement can be directly performed between the
    // two memories.  To illustrate how this works we examine
    // all the memories visible to our local processor in 
    // this mapper.  We can get our set of visible memories
    // using the 'get_visible_memories' method on the machine.
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
const std::map<VariantID,Processor::Kind>& AdversarialMapper::find_task_variants(
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
void AdversarialMapper::map_constrained_requirement(MapperContext ctx,
    const RegionRequirement &req, MappingKind mapping_kind, 
    const std::vector<LayoutConstraintID> &constraints,
    std::vector<PhysicalInstance> &chosen_instances, Processor restricted) 
//--------------------------------------------------------------------------
{
  chosen_instances.resize(constraints.size());
  unsigned output_idx = 0;
  for (std::vector<LayoutConstraintID>::const_iterator lay_it = 
        constraints.begin(); lay_it != 
        constraints.end(); lay_it++, output_idx++)
  {
    const LayoutConstraintSet &layout_constraints = 
      runtime->find_layout_constraints(ctx, *lay_it);
    // TODO: explore the constraints in more detail and exploit randomness
    // We'll use the default mapper to fill in any constraint holes for now
    Machine::MemoryQuery all_memories(machine);
    if (restricted.exists())
      all_memories.has_affinity_to(restricted);
    // This could be a big data structure in a big machine
    std::map<unsigned,Memory> random_memories;
    for (Machine::MemoryQuery::iterator it = all_memories.begin();
          it != all_memories.end(); it++)
    {
      random_memories[generate_random_integer()] = *it;
    }
    bool made_instance = false;
    while (!random_memories.empty())
    {
      std::map<unsigned,Memory>::iterator it = random_memories.begin();
      Memory target = it->second;
      random_memories.erase(it);
      if (target.capacity() == 0)
        continue;
      if (default_make_instance(ctx, target, layout_constraints,
            chosen_instances[output_idx], mapping_kind, 
            true/*force new*/, false/*meets*/, req))
      {
        made_instance = true;
        break;
      }
    }
    if (!made_instance)
    {
      printf("Test mapper %s ran out of memory",
                            get_mapper_name());
      assert(false);
    }
  }
}

//--------------------------------------------------------------------------
void AdversarialMapper::map_random_requirement(MapperContext ctx,
    const RegionRequirement &req, 
    std::vector<PhysicalInstance> &chosen_instances, Processor restricted)
//--------------------------------------------------------------------------
{
  
  std::vector<LogicalRegion> regions(1, req.region);
  chosen_instances.resize(req.privilege_fields.size());
  unsigned output_idx = 0;
  // Iterate over all the fields and make a separate instance and
  // put it in random places
  for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
        it != req.privilege_fields.end(); it++, output_idx++)
  {
    std::vector<FieldID> field(1, *it);
    // Try a bunch of memories in a random order until we find one 
    // that succeeds
    Machine::MemoryQuery all_memories(machine);
    if (restricted.exists())
      all_memories.has_affinity_to(restricted);
    // This could be a big data structure in a big machine
    std::map<unsigned,Memory> random_memories;
    for (Machine::MemoryQuery::iterator it = all_memories.begin();
          it != all_memories.end(); it++)
    {
      random_memories[generate_random_integer()] = *it;
    }
    bool made_instance = false;
    while (!random_memories.empty())
    {
      std::map<unsigned,Memory>::iterator it = random_memories.begin();
      Memory target = it->second;
      random_memories.erase(it);
      if (target.capacity() == 0)
        continue;
      // TODO: put in arbitrary constraints to mess with the DMA system
      LayoutConstraintSet constraints;
      default_policy_select_constraints(ctx, constraints, target, req);
      // Overwrite the field constraints 
      constraints.field_constraint = FieldConstraint(field, false);
      // Try to make the instance, we always make new instances to
      // generate as much data movement and dependence analysis as
      // we possibly can, it will also stress the garbage collector
      if (runtime->create_physical_instance(ctx, target, constraints,
                               regions, chosen_instances[output_idx]))
      {
        made_instance = true;
        break;
      }
    }
    if (!made_instance)
    {
      printf("Test mapper %s ran out of memory",
                            get_mapper_name());
      assert(false);
    }
  }
}

// The next mapping call that we override is the map_task
// mapper method. Once a task has been assigned to map on
// a specific processor (the target_proc) then this method
// is invoked by the runtime to select the memories in 
// which to create physical instances for each logical region.
// The mapper communicates this information to the runtime
// via the mapping fields on RegionRequirements. The memories
// containing currently valid physical instances for each
// logical region is provided by the runtime in the 
// 'current_instances' field. The mapper must specify an
// ordered list of memories for the runtime to try when
// creating a physical instance in the 'target_ranking'
// vector field of each RegionRequirement. The runtime
// attempts to find or make a physical instance in each 
// memory until it succeeds. If the runtime fails to find
// or make a physical instance in any of the memories, then
// the mapping fails and the mapper will be notified that
// the task failed to map using the 'notify_mapping_failed'
// mapper call. If the mapper does nothing, then the task
// is placed back on the list of tasks eligible to be mapped.
// There are other fields that the mapper can set in the
// process of the map_task call that we do not cover here.
//
// In this example, the mapper finds the set of all visible
// memories from the target processor and then puts them
// in a random order as the target set of memories, thereby
// challenging the Legion runtime to maintain correctness
// of data moved through random sets of memories.
int splitable = 1;
void AdversarialMapper::map_task(const MapperContext         ctx,
                                 const Task&                 task,
                                 const MapTaskInput&         input,
                                       MapTaskOutput&        output)
{
  // Pick a random variant, then pick separate instances for all the 
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
  if (task.recursiveable) {
    printf("!!!TASK recuraiveable\n");
  }
  if (variants.size() > 1)
  {
    int chosen_v = 0;
    printf("number of variants %d\n", variants.size());
    if (task.task_id == 2) {
      const int point = task.index_point.point_data[0];
      if (point == 2 && splitable) {
        chosen_v = 1;
        splitable = 0;
      } else {
        chosen_v = 0;
      }
      output.chosen_variant = variants[chosen_v];
    }
    
  }
  else
    output.chosen_variant = variants[0];
  
#if 0
  output.target_procs.push_back(task.target_proc);
  std::vector<bool> premapped(task.regions.size(), false);
  for (unsigned idx = 0; idx < input.premapped_regions.size(); idx++)
  {
    unsigned index = input.premapped_regions[idx];
    output.chosen_instances[index] = input.valid_instances[index];
    premapped[index] = true;
  }
  // Get the execution layout constraints for this variant
  const TaskLayoutConstraintSet &layout_constraints = 
    runtime->find_task_layout_constraints(ctx, task.task_id, 
                                          output.chosen_variant);
  for (unsigned idx = 0; idx < task.regions.size(); idx++)
  {
    if (premapped[idx])
      continue;
    if (task.regions[idx].is_restricted())
    {
      output.chosen_instances[idx] = input.valid_instances[idx];
      continue;
    }
    // See if we have any layout constraints for this index
    // If we do we have to follow them, otherwise we can 
    // let all hell break loose and do what we want
    if (layout_constraints.layouts.find(idx) != 
          layout_constraints.layouts.end())
    {
      std::vector<LayoutConstraintID> constraints;
      for (std::multimap<unsigned,LayoutConstraintID>::const_iterator it = 
            layout_constraints.layouts.lower_bound(idx); it !=
            layout_constraints.layouts.upper_bound(idx); it++)
        constraints.push_back(it->second);
      map_constrained_requirement(ctx, task.regions[idx], TASK_MAPPING,
          constraints, output.chosen_instances[idx], task.target_proc);
    }
    else
      map_random_requirement(ctx, task.regions[idx], 
                             output.chosen_instances[idx],
                             task.target_proc);
  }
  // Give it a random priority
  output.task_priority = default_generate_random_integer();
#endif
  
 // Processor::Kind target_kind = task.target_proc.kind();

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


  // Finally, let's ask for some profiling data to see the impact of our choices
  {
    using namespace ProfilingMeasurements;
    output.task_prof_requests.add_measurement<OperationStatus>();
    output.task_prof_requests.add_measurement<OperationTimeline>();
    output.task_prof_requests.add_measurement<RuntimeOverhead>();
  }
}

void AdversarialMapper::report_profiling(const MapperContext      ctx,
					 const Task&              task,
					 const TaskProfilingInfo& input)
{
  // Local import of measurement names saves typing here without polluting
  // namespace for everybody else
  using namespace ProfilingMeasurements;

  // You are not guaranteed to get measurements you asked for, so make sure to
  // check the result of calls to get_measurement (or just call has_measurement
  // first).  Also, the call returns a copy of the result that you must delete
  // yourself.
  OperationStatus *status = 
    input.profiling_responses.get_measurement<OperationStatus>();
  if (status)
  {
    switch (status->result)
    {
      case OperationStatus::COMPLETED_SUCCESSFULLY:
        {
          printf("Task %s COMPLETED SUCCESSFULLY\n", task.get_task_name());
          break;
        }
      case OperationStatus::COMPLETED_WITH_ERRORS:
        {
          printf("Task %s COMPLETED WITH ERRORS\n", task.get_task_name());
          break;
        }
      case OperationStatus::INTERRUPT_REQUESTED:
        {
          printf("Task %s was INTERRUPTED\n", task.get_task_name());
          break;
        }
      case OperationStatus::TERMINATED_EARLY:
        {
          printf("Task %s TERMINATED EARLY\n", task.get_task_name());
          break;
        }
      case OperationStatus::CANCELLED:
        {
          printf("Task %s was CANCELLED\n", task.get_task_name());
          break;
        }
      default:
        assert(false); // shouldn't get any of the rest currently
    }
    delete status;
  }
  else
    printf("No operation status for task %s\n", task.get_task_name());

  OperationTimeline *timeline =
    input.profiling_responses.get_measurement<OperationTimeline>();
  if (timeline)
  {
    printf("Operation timeline for task %s: ready=%lld start=%lld stop=%lld\n",
	   task.get_task_name(),
	   timeline->ready_time,
	   timeline->start_time,
	   timeline->end_time);
    delete timeline;
  }
  else
    printf("No operation timeline for task %s\n", task.get_task_name());

  RuntimeOverhead *overhead =
    input.profiling_responses.get_measurement<RuntimeOverhead>();
  if (overhead)
  {
    long long total = (overhead->application_time +
		       overhead->runtime_time +
		       overhead->wait_time);
    if (total <= 0) total = 1;
    printf("Runtime overhead for task %s: runtime=%.1f%% wait=%.1f%%\n",
	   task.get_task_name(),
	   (100.0 * overhead->runtime_time / total),
	   (100.0 * overhead->wait_time / total));
    delete overhead;
  }
  else
    printf("No runtime overhead data for task %s\n", task.get_task_name()); 
}

#if 0
PartitioningMapper::PartitioningMapper(Machine m,
                                       Runtime *rt,
                                       Processor p)
  : DefaultMapper(rt->get_mapper_runtime(), m, p)
{
}

void PartitioningMapper::select_tunable_value(const MapperContext ctx,
                                              const Task& task,
                                              const SelectTunableInput& input,
                                                    SelectTunableOutput& output)
{
  if (input.tunable_id == SUBREGION_TUNABLE)
  {
    Machine::ProcessorQuery all_procs(machine);
    all_procs.only_kind(Processor::LOC_PROC);
    runtime->pack_tunable<size_t>(all_procs.count(), output);
    return;
  }
  // Should never get here
  assert(false);
}
#endif

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
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
    }
  }
  int num_subregions = 4;

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
  IndexLauncher daxpy_launcher(DAXPY_TASK_ID, color_is,
                TaskArgument(&alpha, sizeof(alpha)), arg_map);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, input_lr));
  daxpy_launcher.add_field(0, FID_X);
  daxpy_launcher.add_field(0, FID_Y);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, output_lr));
  daxpy_launcher.add_field(1, FID_Z);
  daxpy_launcher.set_recursiveable(true);
  runtime->execute_index_space(ctx, daxpy_launcher);
                    
  TaskLauncher check_launcher(CHECK_TASK_ID, TaskArgument(&alpha, sizeof(alpha)));
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
  assert(task->arglen == sizeof(double));
  const double alpha = *((const double*)task->args);
  const int point = task->index_point.point_data[0];

  const FieldAccessor<READ_ONLY,double,1> acc_x(regions[0], FID_X);
  const FieldAccessor<READ_ONLY,double,1> acc_y(regions[0], FID_Y);
  const FieldAccessor<WRITE_DISCARD,double,1> acc_z(regions[1], FID_Z);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
  task->regions[0].region.get_index_space());
  int size = rect.hi.x - rect.lo.x + 1;

  printf("Running daxpy computation with alpha %.8g for point %d size %d...\n", 
          alpha, point, size);

  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc_z[*pir] = alpha * acc_x[*pir] + acc_y[*pir];
  
}

void daxpy_task_split(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(double));
  const double alpha = *((const double*)task->args);
  const int point = task->index_point.point_data[0];

  Rect<1> rect = runtime->get_index_space_domain(ctx,
  task->regions[0].region.get_index_space());
  int size = rect.hi.x - rect.lo.x + 1;

  printf("Running daxpy split with alpha %.8g for point %d size %d...\n", 
          alpha, point, size);
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
  IndexLauncher daxpy_launcher(DAXPY_TASK_ID, color_is,
                TaskArgument(&alpha, sizeof(alpha)), arg_map);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, input_lr));
  daxpy_launcher.region_requirements[0].add_field(FID_X);
  daxpy_launcher.region_requirements[0].add_field(FID_Y);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, output_lr));
  daxpy_launcher.region_requirements[1].add_field(FID_Z);
  runtime->execute_index_space(ctx, daxpy_launcher);
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(double));
  const double alpha = *((const double*)task->args);

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
