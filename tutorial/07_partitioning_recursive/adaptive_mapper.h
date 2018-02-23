#ifndef __ADAPTIVE_MAPPER_H__
#define __ADAPTIVE_MAPPER_H__

#include "legion.h"
#include "default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

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

class AdaptiveMapper : public DefaultMapper {
public:
  AdaptiveMapper(Machine machine, 
      Runtime *rt, Processor local, RecursiveTaskMapperShared *shared);
public:
	MapperSyncModel get_mapper_sync_model(void) const {
	    return SERIALIZED_NON_REENTRANT_MAPPER_MODEL;
	}
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
	bool select_tasks_to_map_relocate;
  bool slow_down_mapper;
  
  int num_ready_tasks;
  int min_ready_tasks_to_enable_steal;
  
};

void mapper_registration(Machine machine, Runtime *rt,
                         const std::set<Processor> &local_procs);


#endif