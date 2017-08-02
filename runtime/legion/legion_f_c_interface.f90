module legion_fortran_c_interface
    use, intrinsic :: iso_c_binding
    use legion_fortran_types
    implicit none
    
    interface
        ! -----------------------------------------------------------------------
        ! Start-up Operations
        ! -----------------------------------------------------------------------
        ! Legion::Runtime::set_top_level_task_id()
        subroutine legion_runtime_set_top_level_task_id_f(top_id) &
                       bind(C, name="legion_runtime_set_top_level_task_id")
            use iso_c_binding
            implicit none
        
            integer(c_int), value, intent(in) :: top_id
        end subroutine legion_runtime_set_top_level_task_id_f
    
        ! Legion::ExecutionConstraintSet::ExecutionConstraintSet()
        function legion_execution_constraint_set_create_f() &
                     bind(C, name="legion_execution_constraint_set_create")
            use iso_c_binding
            import legion_execution_constraint_set_f_t
            implicit none
        
            type(legion_execution_constraint_set_f_t) :: legion_execution_constraint_set_create_f
        end function
    
        ! Legion::ExecutionConstraintSet::add_constraint(Legion::ProcessorConstraint)
        subroutine legion_execution_constraint_set_add_processor_constraint_f(handle, proc_kind) &
                       bind(C, name="legion_execution_constraint_set_add_processor_constraint")
            use iso_c_binding
            import legion_execution_constraint_set_f_t
            implicit none
        
            type(legion_execution_constraint_set_f_t), value, intent(in)    :: handle
            integer(c_int), value, intent(in)                               :: proc_kind
        end subroutine
    
        ! Legion::TaskLayoutConstraintSet::TaskLayoutConstraintSet()
        function legion_task_layout_constraint_set_create_f() &
                     bind(C, name="legion_task_layout_constraint_set_create")
            use iso_c_binding
            import legion_task_layout_constraint_set_f_t
            implicit none
        
            type(legion_task_layout_constraint_set_f_t) :: legion_task_layout_constraint_set_create_f
        end function
    
        ! Legion::Runtime::preregister_task_variant()
        function legion_runtime_preregister_task_variant_fnptr_f(id, task_name, &
                                                                 execution_constraints, &
                                                                 layout_constraints, &
                                                                 options, &
                                                                 wrapped_task_pointer, &
                                                                 userdata, &
                                                                 userlen) &
                     bind(C, name="legion_runtime_preregister_task_variant_fnptr")
            use iso_c_binding
            import legion_execution_constraint_set_f_t
            import legion_task_layout_constraint_set_f_t
            import legion_task_config_options_f_t
            implicit none
        
            integer(c_int)                                                  :: legion_runtime_preregister_task_variant_fnptr_f
            character(kind=c_char), intent(in)                              :: task_name(*)
            integer(c_int), value, intent(in)                               :: id
            type(legion_execution_constraint_set_f_t), value, intent(in)    :: execution_constraints
            type(legion_task_layout_constraint_set_f_t), value, intent(in)  :: layout_constraints
            type(legion_task_config_options_f_t), value, intent(in)         :: options
            type(c_funptr), value, intent(in)                               :: wrapped_task_pointer
            type(c_ptr), value, intent(in)                                  :: userdata
            integer(c_size_t), value, intent(in)                            :: userlen
        end function
    
        ! Legion::Runtime::start()
        function legion_runtime_start_f(argc, argv, background) &
                     bind(C, name="legion_runtime_start")
            use iso_c_binding
            implicit none
        
            integer(c_int)                      :: legion_runtime_start_f
            integer(c_int), value, intent(in)   :: argc
            type(c_ptr), value, intent(in)      :: argv
            logical(c_bool), value, intent(in)  :: background
        end function
    
        ! Legion::LegionTaskWrapper::legion_task_preamble()
        subroutine legion_task_preamble_f(tdata, tdatalen, proc_id, &
                                          task, regionptr, num_regions, &
                                          ctx, runtime) &
                       bind(C, name="legion_task_preamble")
            use iso_c_binding
            import legion_task_f_t
            import legion_physical_region_f_t
            import legion_context_f_t
            import legion_runtime_f_t
            implicit none
        
            type(c_ptr), intent(in)                         :: tdata ! pass reference
            integer(c_size_t), value, intent(in)            :: tdatalen
            integer(c_long_long), value, intent(in)         :: proc_id
            type(legion_task_f_t), intent(out)              :: task ! pass reference
            type(c_ptr), intent(out)                        :: regionptr
            integer(c_int), intent(out)                     :: num_regions ! pass reference
            type(legion_context_f_t), intent(out)           :: ctx ! pass reference          
            type(legion_runtime_f_t), intent(out)           :: runtime ! pass reference
        end subroutine
    
        ! Legion::LegionTaskWrapper::legion_task_postamble()
        subroutine legion_task_postamble_f(runtime, ctx, retval, retsize) &
                       bind(C, name="legion_task_postamble")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_context_f_t
            implicit none
        
            type(legion_runtime_f_t), value, intent(in) :: runtime
            type(legion_context_f_t), value, intent(in) :: ctx
            type(c_ptr), value, intent(in)              :: retval
            integer(c_size_t), value, intent(in)        :: retsize
        end subroutine
    
        ! -----------------------------------------------------------------------
        ! Task Launcher
        ! -----------------------------------------------------------------------
        ! @see Legion::TaskLauncher::TaskLauncher()
        function legion_task_launcher_create_c(tid, arg, pred, id, tag) &
                bind(C, name="legion_task_launcher_create")
            use iso_c_binding
            import legion_task_launcher_f_t
            import legion_task_argument_f_t
            import legion_predicate_f_t
            implicit none
        
            type(legion_task_launcher_f_t)                      :: legion_task_launcher_create_c
            integer(c_int), value, intent(in)                   :: tid
            type(legion_task_argument_f_t), value, intent(in)   :: arg
            type(legion_predicate_f_t), value, intent(in)       :: pred
            integer(c_int), value, intent(in)                   :: id
            integer(c_long), value, intent(in)                  :: tag
        end function legion_task_launcher_create_c
    
        ! @see Legion::TaskLauncher::~TaskLauncher()
        subroutine legion_task_launcher_destroy_c(handle) &
                bind(C, name="legion_task_launcher_destroy")
            use iso_c_binding
            import legion_task_launcher_f_t
            implicit none
        
            type(legion_task_launcher_f_t), value, intent(in) :: handle
        end subroutine legion_task_launcher_destroy_c
    
        ! @see Legion::Runtime::execute_task()
        function legion_task_launcher_execute_c(runtime, ctx, launcher) &
                bind(C, name="legion_task_launcher_execute")
            use iso_c_binding
            import legion_future_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_task_launcher_f_t
            implicit none
        
            type(legion_future_f_t)                            :: legion_task_launcher_execute_c
            type(legion_runtime_f_t), value, intent(in)        :: runtime
            type(legion_context_f_t), value, intent(in)        :: ctx
            type(legion_task_launcher_f_t), value, intent(in)  :: launcher
        end function legion_task_launcher_execute_c
    
        ! @see Legion::TaskLauncher::add_region_requirement()
        function legion_task_launcher_add_region_requirement_logical_region_c(launcher, handle, priv, prop, parent, tag, verified) &
                bind (C, name="legion_task_launcher_add_region_requirement_logical_region")
            use iso_c_binding
            import legion_task_launcher_f_t
            import legion_logical_region_f_t
            implicit none
        
            integer(c_int)                                     :: legion_task_launcher_add_region_requirement_logical_region_c
            type(legion_task_launcher_f_t), value, intent(in)  :: launcher
            type(legion_logical_region_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)                  :: priv
            integer(c_int), value, intent(in)                  :: prop
            type(legion_logical_region_f_t), value, intent(in) :: parent
            integer(c_long), value, intent(in)                 :: tag
            logical(c_bool), value, intent(in)                 :: verified
        end function legion_task_launcher_add_region_requirement_logical_region_c
    
        ! @see Legion::TaskLaunchxer::add_field()
        subroutine legion_task_launcher_add_field_c(launcher, idx, fid, inst) &
                bind(C, name="legion_task_launcher_add_field")
            use iso_c_binding
            import legion_task_launcher_f_t
            implicit none
        
            type(legion_task_launcher_f_t), value, intent(in) :: launcher
            integer(c_int), value, intent(in)                 :: idx
            integer(c_int), value, intent(in)                 :: fid
            logical(c_bool), value, intent(in)                :: inst 
        end subroutine legion_task_launcher_add_field_c
    
        ! -----------------------------------------------------------------------
        ! Index Launcher
        ! -----------------------------------------------------------------------
        ! @see Legion::IndexTaskLauncher::IndexTaskLauncher()
        function legion_index_launcher_create_c(tid, domain, global_arg, map, pred, must, id, tag) &
                bind(C, name="legion_index_launcher_create")
            use iso_c_binding
            import legion_index_launcher_f_t
            import legion_domain_f_t
            import legion_task_argument_f_t
            import legion_argument_map_f_t
            import legion_predicate_f_t
            implicit none
        
            type(legion_index_launcher_f_t)                     :: legion_index_launcher_create_c
            integer(c_int), value, intent(in)                   :: tid
            type(legion_domain_f_t), value, intent(in)          :: domain
            type(legion_task_argument_f_t), value, intent(in)   :: global_arg
            type(legion_argument_map_f_t), value, intent(in)    :: map
            type(legion_predicate_f_t), value, intent(in)       :: pred
            logical(c_bool), value, intent(in)                  :: must
            integer(c_int), value, intent(in)                   :: id
            integer(c_long), value, intent(in)                  :: tag
        end function legion_index_launcher_create_c
    
        ! @see Legion::IndexTaskLauncher::~IndexTaskLauncher()
        subroutine legion_index_launcher_destroy_c(handle) &
                bind(C, name="legion_index_launcher_destroy")
            use iso_c_binding
            import legion_index_launcher_f_t
            implicit none
        
            type(legion_index_launcher_f_t), value, intent(in) :: handle
        end subroutine legion_index_launcher_destroy_c
    
        ! @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &)
        function legion_index_launcher_execute_c(runtime, ctx, launcher) &
                bind(C, name="legion_index_launcher_execute")
            use iso_c_binding
            import legion_future_map_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_index_launcher_f_t
            implicit none
        
            type(legion_future_map_f_t)                         :: legion_index_launcher_execute_c
            type(legion_runtime_f_t), value, intent(in)         :: runtime
            type(legion_context_f_t), value, intent(in)         :: ctx
            type(legion_index_launcher_f_t), value, intent(in)  :: launcher
        end function legion_index_launcher_execute_c
    
        ! @see Legion::Runtime::execute_index_space(Context, const IndexTaskLauncher &, ReductionOpID)
        function legion_index_launcher_execute_reduction_c(runtime, ctx, launcher, redop) &
                bind(C, name="legion_index_launcher_execute_reduction")
            use iso_c_binding
            import legion_future_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_index_launcher_f_t
            implicit none
        
            type(legion_future_f_t)                            :: legion_index_launcher_execute_reduction_c
            type(legion_runtime_f_t), value, intent(in)        :: runtime
            type(legion_context_f_t), value, intent(in)        :: ctx
            type(legion_index_launcher_f_t), value, intent(in) :: launcher
            integer(c_int), value, intent(in)                  :: redop 
        end function legion_index_launcher_execute_reduction_c
    
        ! @see Legion::IndexTaskLauncher::add_region_requirement()
        function legion_index_launcher_add_region_requirement_lp_c(launcher, handle, proj, priv, &
                prop, parent, tag, verified) &
                bind (C, name="legion_index_launcher_add_region_requirement_logical_partition")
            use iso_c_binding
            import legion_index_launcher_f_t
            import legion_logical_partition_f_t
            import legion_logical_region_f_t
            implicit none
        
            integer(c_int)                              :: legion_index_launcher_add_region_requirement_lp_c
            type(legion_index_launcher_f_t), value, intent(in)    :: launcher
            type(legion_logical_partition_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)                     :: proj
            integer(c_int), value, intent(in)                     :: priv
            integer(c_int), value, intent(in)                     :: prop
            type(legion_logical_region_f_t), value, intent(in)    :: parent
            integer(c_long), value, intent(in)                    :: tag
            logical(c_bool), value, intent(in)                    :: verified
        end function legion_index_launcher_add_region_requirement_lp_c
    
        ! @see Legion::TaskLaunchxer::add_field()
        subroutine legion_index_launcher_add_field_c(launcher, idx, fid, inst) &
                bind(C, name="legion_index_launcher_add_field")
            use iso_c_binding
            import legion_index_launcher_f_t
            implicit none
        
            type(legion_index_launcher_f_t), value, intent(in) :: launcher
            integer(c_int), value, intent(in)                  :: idx
            integer(c_int), value, intent(in)                  :: fid
            logical(c_bool), value, intent(in)                 :: inst 
        end subroutine legion_index_launcher_add_field_c
    
        ! -----------------------------------------------------------------------
        ! Predicate Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Predicate::TRUE_PRED
        function legion_predicate_true_c() &
                     bind(C, name="legion_predicate_true")
            use iso_c_binding
            import legion_predicate_f_t
            implicit none
        
            type(legion_predicate_f_t)  :: legion_predicate_true_c
        end function legion_predicate_true_c
    
        ! @see Legion::Predicate::FALSE_PRED
        function legion_predicate_false_c() &
                     bind(C, name="legion_predicate_false")
            use iso_c_binding
            import legion_predicate_f_t
            implicit none
        
            type(legion_predicate_f_t)  :: legion_predicate_false_c
        end function legion_predicate_false_c
    
        ! -----------------------------------------------------------------------
        ! Argument Map
        ! -----------------------------------------------------------------------
        ! @see Legion::ArgumentMap::ArgumentMap()
        function legion_argument_map_create_c() &
                bind(C, name="legion_argument_map_create")
            use iso_c_binding
            import legion_argument_map_f_t
            implicit none
        
            type(legion_argument_map_f_t) :: legion_argument_map_create_c
        end function legion_argument_map_create_c
    
        ! @see Legion::ArgumentMap::set_point()
        subroutine legion_argument_map_set_point_c(map, dp, arg, replace) &
                bind(C, name="legion_argument_map_set_point")
            use iso_c_binding
            import legion_argument_map_f_t
            import legion_domain_point_f_t
            import legion_task_argument_f_t
            implicit none
        
            type(legion_argument_map_f_t), value, intent(in)  :: map
            type(legion_domain_point_f_t), value, intent(in)  :: dp 
            type(legion_task_argument_f_t), value, intent(in) :: arg
            logical(c_bool), value, intent(in)                :: replace 
        end subroutine legion_argument_map_set_point_c
    
        ! -----------------------------------------------------------------------
        ! Task Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Task::args
        function legion_task_get_args_c(task) &
                bind(C, name="legion_task_get_args")
            use iso_c_binding
            import legion_task_f_t
            implicit none
        
            type(c_ptr)                              :: legion_task_get_args_c
            type(legion_task_f_t), value, intent(in) :: task
        end function legion_task_get_args_c
    
        ! @see Legion::Task::arglen
        function legion_task_get_arglen_c(task) &
                bind(C, name="legion_task_get_arglen")
            use iso_c_binding
            import legion_task_f_t
            implicit none
        
            integer(c_size_t)                        :: legion_task_get_arglen_c
            type(legion_task_f_t), value, intent(in) :: task
        end function legion_task_get_arglen_c
    
        ! @see Legion::Task::local_args
        function legion_task_get_local_args_c(task) &
                bind(C, name="legion_task_get_local_args")
            use iso_c_binding
            import legion_task_f_t
            implicit none
        
            type(c_ptr)                              :: legion_task_get_local_args_c
            type(legion_task_f_t), value, intent(in) :: task
        end function legion_task_get_local_args_c
    
        ! @see Legion::Task::local_arglen
        function legion_task_get_local_arglen_c(task) &
                bind(C, name="legion_task_get_local_arglen")
            use iso_c_binding
            import legion_task_f_t
            implicit none
        
            integer(c_size_t)                        :: legion_task_get_local_arglen_c
            type(legion_task_f_t), value, intent(in) :: task
        end function legion_task_get_local_arglen_c
    
        ! @see Legion::Task::index_domain
        function legion_task_get_index_domain_c(task) &
                bind(C, name="legion_task_get_index_domain")
            use iso_c_binding
            import legion_domain_f_t
            import legion_task_f_t
            implicit none
        
            type(legion_domain_f_t)                  :: legion_task_get_index_domain_c
            type(legion_task_f_t), value, intent(in) :: task
        end function legion_task_get_index_domain_c
    
        ! @see Legion::Task::regions
        function legion_task_get_region_c(task, idx) &
                bind(C, name="legion_task_get_region")
            use iso_c_binding
            import legion_region_requirement_f_t
            import legion_task_f_t
            implicit none
        
            type(legion_region_requirement_f_t)        :: legion_task_get_region_c
            type(legion_task_f_t), value, intent(in) :: task
            integer(c_int), value, intent(in)        :: idx
        end function legion_task_get_region_c
    
        ! -----------------------------------------------------------------------
        ! Domain Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Domain::from_rect()
        function legion_domain_from_rect_1d_c(r) &
                bind(C, name="legion_domain_from_rect_1d")
            use iso_c_binding
            import legion_rect_1d_f_t
            import legion_domain_f_t
            implicit none
        
            type(legion_domain_f_t)                     :: legion_domain_from_rect_1d_c
            type(legion_rect_1d_f_t), value, intent(in) :: r
        end function legion_domain_from_rect_1d_c
    
        ! @see Legion::Domain::from_rect()
        function legion_domain_from_rect_2d_c(r) &
                bind(C, name="legion_domain_from_rect_2d")
            use iso_c_binding
            import legion_rect_2d_f_t
            import legion_domain_f_t
            implicit none
        
            type(legion_domain_f_t)                     :: legion_domain_from_rect_2d_c
            type(legion_rect_2d_f_t), value, intent(in) :: r
        end function legion_domain_from_rect_2d_c
    
        ! @see Legion::Domain::from_rect()
        function legion_domain_from_rect_3d_c(r) &
                bind(C, name="legion_domain_from_rect_3d")
            use iso_c_binding
            import legion_rect_3d_f_t
            import legion_domain_f_t
            implicit none
        
            type(legion_domain_f_t)                     :: legion_domain_from_rect_3d_c
            type(legion_rect_3d_f_t), value, intent(in) :: r
        end function legion_domain_from_rect_3d_c
    
        ! @see Legion::Domain::get_rect()
        function legion_domain_get_rect_1d_c(d) &
                bind(C, name="legion_domain_get_rect_1d")
            use iso_c_binding
            import legion_rect_1d_f_t
            import legion_domain_f_t
            implicit none
        
            type(legion_rect_1d_f_t)                   :: legion_domain_get_rect_1d_c
            type(legion_domain_f_t), value, intent(in) :: d
        end function legion_domain_get_rect_1d_c
    
        ! @see Legion::Domain::get_rect()
        function legion_domain_get_rect_2d_c(d) &
                bind(C, name="legion_domain_get_rect_2d")
            use iso_c_binding
            import legion_rect_2d_f_t
            import legion_domain_f_t
            implicit none
        
            type(legion_rect_2d_f_t)                   :: legion_domain_get_rect_2d_c
            type(legion_domain_f_t), value, intent(in) :: d
        end function legion_domain_get_rect_2d_c
    
        ! @see Legion::Domain::get_rect()
        function legion_domain_get_rect_3d_c(d) &
                bind(C, name="legion_domain_get_rect_3d")
            use iso_c_binding
            import legion_rect_3d_f_t
            import legion_domain_f_t
            implicit none
        
            type(legion_rect_3d_f_t)                   :: legion_domain_get_rect_3d_c
            type(legion_domain_f_t), value, intent(in) :: d
        end function legion_domain_get_rect_3d_c
    
        ! @see Legion::Domain::get_volume()
        function legion_domain_get_volume_c(d) &
                bind(C, name="legion_domain_get_volume")
            use iso_c_binding
            import legion_domain_f_t
            implicit none
        
            integer(c_size_t)                          :: legion_domain_get_volume_c
            type(legion_domain_f_t), value, intent(in) :: d
        end function legion_domain_get_volume_c
    
        ! -----------------------------------------------------------------------
        ! Domain Point Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Domain::from_point()
        function legion_domain_point_from_point_1d_c(p) &
                bind(C, name="legion_domain_point_from_point_1d")
            use iso_c_binding
            import legion_domain_point_f_t
            import legion_point_1d_f_t
            implicit none
        
            type(legion_domain_point_f_t)                :: legion_domain_point_from_point_1d_c
            type(legion_point_1d_f_t), value, intent(in) :: p
        end function legion_domain_point_from_point_1d_c
    
        ! @see Legion::Domain::from_point()
        function legion_domain_point_from_point_2d_c(p) &
                bind(C, name="legion_domain_point_from_point_2d")
            use iso_c_binding
            import legion_domain_point_f_t
            import legion_point_2d_f_t
            implicit none
        
            type(legion_domain_point_f_t)                :: legion_domain_point_from_point_2d_c
            type(legion_point_2d_f_t), value, intent(in) :: p
        end function legion_domain_point_from_point_2d_c
    
        ! @see Legion::Domain::from_point()
        function legion_domain_point_from_point_3d_c(p) &
                bind(C, name="legion_domain_point_from_point_3d")
            use iso_c_binding
            import legion_domain_point_f_t
            import legion_point_3d_f_t
            implicit none
        
            type(legion_domain_point_f_t)                :: legion_domain_point_from_point_3d_c
            type(legion_point_3d_f_t), value, intent(in) :: p
        end function legion_domain_point_from_point_3d_c
    
        ! -----------------------------------------------------------------------
        ! Future Map Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::FutureMap::wait_all_results()
        subroutine legion_future_map_wait_all_results_c(handle) &
                bind(C, name="legion_future_map_wait_all_results")
            use iso_c_binding
            import legion_future_map_f_t
            implicit none
        
            type(legion_future_map_f_t), value, intent(in) :: handle
        end subroutine legion_future_map_wait_all_results_c
    
        ! -----------------------------------------------------------------------
        ! Index Space Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Runtime::create_index_space(Context, Domain)
        function legion_index_space_create_domain_c(runtime, ctx, domain) &
                bind(C, name="legion_index_space_create_domain")
            use iso_c_binding
            import legion_index_space_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_domain_f_t
            implicit none
        
            type(legion_index_space_f_t)                :: legion_index_space_create_domain_c
            type(legion_runtime_f_t), value, intent(in) :: runtime
            type(legion_context_f_t), value, intent(in) :: ctx
            type(legion_domain_f_t), value, intent(in)  :: domain
        end function legion_index_space_create_domain_c
    
        ! @see Legion::Runtime::destroy_index_space()
        subroutine legion_index_space_destroy_c(runtime, ctx, handle) &
                bind(C, name="legion_index_space_destroy")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_index_space_f_t
            implicit none
        
            type(legion_runtime_f_t), value, intent(in)      :: runtime
            type(legion_context_f_t), value, intent(in)      :: ctx
            type(legion_index_space_f_t), value, intent(in)  :: handle
        end subroutine legion_index_space_destroy_c
    
        ! @see Legion::Runtime::get_index_space_domain()
        function legion_index_space_get_domain_c(runtime, handle) &
                bind(C, name="legion_index_space_get_domain")
            use iso_c_binding
            import legion_domain_f_t
            import legion_runtime_f_t
            import legion_index_space_f_t
            implicit none
        
            type(legion_domain_f_t)                         :: legion_index_space_get_domain_c
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_index_space_f_t), value, intent(in) :: handle
        end function legion_index_space_get_domain_c
    
        ! @see Legion::Runtime::attach_name()
        subroutine legion_index_space_attach_name_c(runtime, handle, name, is_mutable) &
                bind (C, name="legion_index_space_attach_name")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_index_space_f_t
            implicit none
        
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_index_space_f_t), value, intent(in) :: handle
            type(c_ptr), value, intent(in)                 :: name
            logical(c_bool), value, intent(in)              :: is_mutable
        end subroutine legion_index_space_attach_name_c
    
        ! -----------------------------------------------------------------------
        ! Index Partition Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Runtime::create_equal_partition()
        function legion_index_partition_create_equal_c(runtime, ctx, parent, color_space, granularity, color) &
                bind(C, name="legion_index_partition_create_equal")
            use iso_c_binding
            import legion_index_partition_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_index_space_f_t
            implicit none
        
            type(legion_index_partition_f_t)                :: legion_index_partition_create_equal_c
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_context_f_t), value, intent(in)     :: ctx
            type(legion_index_space_f_t), value, intent(in) :: parent
            type(legion_index_space_f_t), value, intent(in) :: color_space
            integer(c_size_t), value, intent(in)            :: granularity
            integer(c_int), value, intent(in)               :: color
        end function legion_index_partition_create_equal_c
    
        ! @see Legion::Runtime::attach_name()
        subroutine legion_index_partition_attach_name_c(runtime, handle, name, is_mutable) &
                bind (C, name="legion_index_partition_attach_name")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_index_partition_f_t
            implicit none
        
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_index_partition_f_t), value, intent(in) :: handle
            type(c_ptr), value, intent(in)                 :: name
            logical(c_bool), value, intent(in)              :: is_mutable
        end subroutine legion_index_partition_attach_name_c
    
        ! -----------------------------------------------------------------------
        ! Logical Region Tree Traversal Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::Runtime::get_logical_partition()
        function legion_logical_partition_create_c(runtime, ctx, parent, handle) &
                bind (C, name="legion_logical_partition_create")
            use iso_c_binding
            import legion_logical_partition_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_logical_region_f_t
            import legion_index_partition_f_t
            implicit none
        
            type(legion_logical_partition_f_t) :: legion_logical_partition_create_c
            type(legion_runtime_f_t), value, intent(in) :: runtime
            type(legion_context_f_t), value, intent(in) :: ctx
            type(legion_logical_region_f_t), value, intent(in) :: parent
            type(legion_index_partition_f_t), value, intent(in) :: handle
        end function legion_logical_partition_create_c
    
        ! @see Legion::Runtime::attach_name()
        subroutine legion_logical_partition_attach_name_c(runtime, handle, name, is_mutable) &
                bind (C, name="legion_logical_partition_attach_name")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_logical_partition_f_t
            implicit none
        
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_logical_partition_f_t), value, intent(in) :: handle
            type(c_ptr), value, intent(in)                 :: name
            logical(c_bool), value, intent(in)              :: is_mutable
        end subroutine legion_logical_partition_attach_name_c
    
        ! -----------------------------------------------------------------------
        ! Field Space Operatiins 
        ! -----------------------------------------------------------------------
        ! @see Legion::Runtime::create_field_space()
        function legion_field_space_create_c(runtime, ctx) &
                bind(C, name="legion_field_space_create")
            use iso_c_binding
            import legion_field_space_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            implicit none
        
            type(legion_field_space_f_t)                :: legion_field_space_create_c
            type(legion_runtime_f_t), value, intent(in) :: runtime
            type(legion_context_f_t), value, intent(in) :: ctx
        end function legion_field_space_create_c
    
        ! @see Legion::Runtime::destroy_field_space()
        subroutine legion_field_space_destroy_c(runtime, ctx, handle) &
                bind(C, name="legion_field_space_destroy")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_field_space_f_t
            implicit none
        
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_context_f_t), value, intent(in)     :: ctx
            type(legion_field_space_f_t), value, intent(in) :: handle
        end subroutine legion_field_space_destroy_c
    
        ! -----------------------------------------------------------------------
        ! Field Allocator 
        ! -----------------------------------------------------------------------
        ! @see Legion::Runtime::create_field_allocator()
        function legion_field_allocator_create_c(runtime, ctx, handle) &
                bind(C, name="legion_field_allocator_create")
            use iso_c_binding
            import legion_field_allocator_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_field_space_f_t
            implicit none
        
            type(legion_field_allocator_f_t)                :: legion_field_allocator_create_c
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_context_f_t), value, intent(in)     :: ctx
            type(legion_field_space_f_t), value, intent(in) :: handle
        end function legion_field_allocator_create_c
    
        ! @see Legion::FieldAllocator::~FieldAllocator()
        subroutine legion_field_allocator_destroy_c(handle) &
                bind(C, name="legion_field_allocator_destroy")
            use iso_c_binding
            import legion_field_allocator_f_t
            implicit none
        
            type(legion_field_allocator_f_t), value, intent(in) :: handle
        end subroutine legion_field_allocator_destroy_c
    
        ! @see Legion::FieldAllocator::allocate_field()
        function legion_field_allocator_allocate_field_c(allocator, field_size, desired_fieldid) &
                bind (C, name="legion_field_allocator_allocate_field")
            use iso_c_binding
            import legion_field_allocator_f_t
            implicit none
        
            integer(c_int)                                      :: legion_field_allocator_allocate_field_c
            type(legion_field_allocator_f_t), value, intent(in) :: allocator                                          
            integer(c_size_t), value, intent(in)                :: field_size                                         
            integer(c_int), value, intent(in)                   :: desired_fieldid                                        
        end function legion_field_allocator_allocate_field_c
    
        ! -----------------------------------------------------------------------
        ! Logical Region
        ! -----------------------------------------------------------------------
        ! @see Legion::Runtime::create_logical_region()
        function legion_logical_region_create_c(runtime, ctx, index, fields) &
                bind(C, name="legion_logical_region_create")
            use iso_c_binding
            import legion_logical_region_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_index_space_f_t
            import legion_field_space_f_t
            implicit none
        
            type(legion_logical_region_f_t)                 :: legion_logical_region_create_c
            type(legion_runtime_f_t), value, intent(in)     :: runtime
            type(legion_context_f_t), value, intent(in)     :: ctx
            type(legion_index_space_f_t), value, intent(in) :: index
            type(legion_field_space_f_t), value, intent(in) :: fields        
        end function legion_logical_region_create_c
    
        ! @see Legion::Runtime::destroy_logical_region()
        subroutine legion_logical_region_destroy_c(runtime, ctx, handle) &
                bind(C, name="legion_logical_region_destroy")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_logical_region_f_t
            implicit none
        
            type(legion_runtime_f_t), value, intent(in)        :: runtime
            type(legion_context_f_t), value, intent(in)        :: ctx
            type(legion_logical_region_f_t), value, intent(in) :: handle
        end subroutine legion_logical_region_destroy_c
    
        ! @see Legion::LogicalRegion::get_index_space
        function legion_logical_region_get_index_space_c(handle) &
                bind(C, name="legion_logical_region_get_index_space")
            use iso_c_binding
            import legion_index_space_f_t
            import legion_logical_region_f_t
            implicit none
        
            type(legion_index_space_f_t)                       :: legion_logical_region_get_index_space_c
            type(legion_logical_region_f_t), value, intent(in) :: handle
        end function legion_logical_region_get_index_space_c
    
        ! -----------------------------------------------------------------------
        ! Region Requirement Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::RegionRequirement::region
        function legion_region_requirement_get_region_c(handle) &
                bind(C, name="legion_region_requirement_get_region")
            use iso_c_binding
            import legion_region_requirement_f_t
            import legion_logical_region_f_t
            implicit none
            
            type(legion_logical_region_f_t)                        :: legion_region_requirement_get_region_c
            type(legion_region_requirement_f_t), value, intent(in) :: handle
        end function legion_region_requirement_get_region_c
        
        ! @see Legion::RegionRequirement::privilege_fields
        function legion_region_requirement_get_privilege_field_c(handle, idx) &
                bind (C, name="legion_region_requirement_get_privilege_field")
            use iso_c_binding
            import legion_region_requirement_f_t
            implicit none
        
            integer(c_int)                                         :: legion_region_requirement_get_privilege_field_c
            type(legion_region_requirement_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)                      :: idx
        end function legion_region_requirement_get_privilege_field_c
    
        ! -----------------------------------------------------------------------
        ! Physical Data Operations
        ! -----------------------------------------------------------------------
        function legion_get_physical_region_by_id_c(regionptr, id, num_regions) &
                bind(C, name="legion_get_physical_region_by_id")
            use iso_c_binding
            import legion_physical_region_f_t
            implicit none
        
            type(legion_physical_region_f_t)  :: legion_get_physical_region_by_id_c
            type(c_ptr), value, intent(in)    :: regionptr
            integer(c_int), value, intent(in) :: id
            integer(c_int), value, intent(in) :: num_regions
        end function legion_get_physical_region_by_id_c
    
        ! @see Legion::PhysicalRegion::get_field_accessor()
        function legion_physical_region_get_field_accessor_array_1d_c(handle, fid) &
                bind(C, name="legion_physical_region_get_field_accessor_array_1d")
            use iso_c_binding
            import legion_accessor_array_1d_f_t
            import legion_physical_region_f_t
            implicit none
        
            type(legion_accessor_array_1d_f_t)                  :: legion_physical_region_get_field_accessor_array_1d_c
            type(legion_physical_region_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)                   :: fid
        end function legion_physical_region_get_field_accessor_array_1d_c
    
        ! @see Legion::PhysicalRegion::get_field_accessor()
        function legion_physical_region_get_field_accessor_array_2d_c(handle, fid) &
                bind(C, name="legion_physical_region_get_field_accessor_array_2d")
            use iso_c_binding
            import legion_accessor_array_2d_f_t
            import legion_physical_region_f_t
            implicit none
        
            type(legion_accessor_array_2d_f_t)                  :: legion_physical_region_get_field_accessor_array_2d_c
            type(legion_physical_region_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)                   :: fid
        end function legion_physical_region_get_field_accessor_array_2d_c
    
        ! @see Legion::PhysicalRegion::get_field_accessor()
        function legion_physical_region_get_field_accessor_array_3d_c(handle, fid) &
                bind(C, name="legion_physical_region_get_field_accessor_array_3d")
            use iso_c_binding
            import legion_accessor_array_3d_f_t
            import legion_physical_region_f_t
            implicit none
        
            type(legion_accessor_array_3d_f_t)                  :: legion_physical_region_get_field_accessor_array_3d_c
            type(legion_physical_region_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)                   :: fid
        end function legion_physical_region_get_field_accessor_array_3d_c
    
        ! @see Legion::UnsafeFieldAccessor::ptr
        function legion_accessor_array_1d_raw_rect_ptr_c(handle, rect, subrect, offset) &
                bind(C, name="legion_accessor_array_1d_raw_rect_ptr")
            use iso_c_binding
            import legion_accessor_array_1d_f_t
            import legion_rect_1d_f_t
            import legion_byte_offset_f_t
            implicit none
        
            type(c_ptr)         :: legion_accessor_array_1d_raw_rect_ptr_c
            type(legion_accessor_array_1d_f_t), value, intent(in) :: handle
            type(legion_rect_1d_f_t), value, intent(in)           :: rect
            type(legion_rect_1d_f_t), intent(out)                 :: subrect ! pass reference
            type(legion_byte_offset_f_t), intent(out)             :: offset  ! pass reference
        end function legion_accessor_array_1d_raw_rect_ptr_c
    
        ! @see Legion::UnsafeFieldAccessor::ptr
        function legion_accessor_array_2d_raw_rect_ptr_c(handle, rect, subrect, offset) &
                bind(C, name="legion_accessor_array_2d_raw_rect_ptr")
            use iso_c_binding
            import legion_accessor_array_2d_f_t
            import legion_rect_2d_f_t
            import legion_byte_offset_f_t
            implicit none
        
            type(c_ptr)         :: legion_accessor_array_2d_raw_rect_ptr_c
            type(legion_accessor_array_2d_f_t), value, intent(in) :: handle
            type(legion_rect_2d_f_t), value, intent(in)           :: rect
            type(legion_rect_2d_f_t), intent(out)                 :: subrect ! pass reference
            type(legion_byte_offset_f_t), intent(out)             :: offset(2)  ! pass reference
        end function legion_accessor_array_2d_raw_rect_ptr_c
    
        ! @see Legion::UnsafeFieldAccessor::ptr
        function legion_accessor_array_3d_raw_rect_ptr_c(handle, rect, subrect, offset) &
                bind(C, name="legion_accessor_array_3d_raw_rect_ptr")
            use iso_c_binding
            import legion_accessor_array_3d_f_t
            import legion_rect_3d_f_t
            import legion_byte_offset_f_t
            implicit none
        
            type(c_ptr)         :: legion_accessor_array_3d_raw_rect_ptr_c
            type(legion_accessor_array_3d_f_t), value, intent(in) :: handle
            type(legion_rect_3d_f_t), value, intent(in)           :: rect
            type(legion_rect_3d_f_t), intent(out)                 :: subrect ! pass reference
            type(legion_byte_offset_f_t), intent(out)             :: offset(3)  ! pass reference
        end function legion_accessor_array_3d_raw_rect_ptr_c
    
        ! @see Legion::UnsafeFieldAccessor::ptr
        subroutine legion_accessor_array_1d_read_point_c(handle, point, dst, bytes) &
                bind(C, name="legion_accessor_array_1d_read_point")
            use iso_c_binding
            import legion_accessor_array_1d_f_t
            import legion_point_1d_f_t
            implicit none
        
            type(legion_accessor_array_1d_f_t), value, intent(in) :: handle
            type(legion_point_1d_f_t), value, intent(in)          :: point
            type(c_ptr), value, intent(in)                        :: dst ! should be OUT, set to IN to cheat compiler
            integer(c_size_t), value, intent(in)                  :: bytes
        end subroutine legion_accessor_array_1d_read_point_c
    
        ! @see Legion::UnsafeFieldAccessor::ptr
        subroutine legion_accessor_array_2d_read_point_c(handle, point, dst, bytes) &
                bind(C, name="legion_accessor_array_2d_read_point")
            use iso_c_binding
            import legion_accessor_array_2d_f_t
            import legion_point_2d_f_t
            implicit none
        
            type(legion_accessor_array_2d_f_t), value, intent(in) :: handle
            type(legion_point_2d_f_t), value, intent(in)          :: point
            type(c_ptr), value, intent(in)                        :: dst ! should be OUT, set to IN to cheat compiler
            integer(c_size_t), value, intent(in)                  :: bytes
        end subroutine legion_accessor_array_2d_read_point_c
    
        ! @see Legion::UnsafeFieldAccessor::ptr
        subroutine legion_accessor_array_3d_read_point_c(handle, point, dst, bytes) &
                bind(C, name="legion_accessor_array_3d_read_point")
            use iso_c_binding
            import legion_accessor_array_3d_f_t
            import legion_point_3d_f_t
            implicit none
        
            type(legion_accessor_array_3d_f_t), value, intent(in) :: handle
            type(legion_point_3d_f_t), value, intent(in)          :: point
            type(c_ptr), value, intent(in)                        :: dst ! should be OUT, set to IN to cheat compiler
            integer(c_size_t), value, intent(in)                  :: bytes
        end subroutine legion_accessor_array_3d_read_point_c
    
        ! @see Legion::UnsafeFieldAccessor::ptr
        subroutine legion_accessor_array_1d_write_point_c(handle, point, src, bytes) &
                bind(C, name="legion_accessor_array_1d_write_point")
            use iso_c_binding
            import legion_accessor_array_1d_f_t
            import legion_point_1d_f_t
            implicit none
        
            type(legion_accessor_array_1d_f_t), value, intent(in) :: handle
            type(legion_point_1d_f_t), value, intent(in)          :: point
            type(c_ptr), value, intent(in)                        :: src
            integer(c_size_t), value, intent(in)                  :: bytes
        end subroutine legion_accessor_array_1d_write_point_c
    
        ! @see Legion::UnsafeFieldAccessor::ptr
        subroutine legion_accessor_array_2d_write_point_c(handle, point, src, bytes) &
                bind(C, name="legion_accessor_array_2d_write_point")
            use iso_c_binding
            import legion_accessor_array_2d_f_t
            import legion_point_2d_f_t
            implicit none
        
            type(legion_accessor_array_2d_f_t), value, intent(in) :: handle
            type(legion_point_2d_f_t), value, intent(in)          :: point
            type(c_ptr), value, intent(in)                        :: src
            integer(c_size_t), value, intent(in)                  :: bytes
        end subroutine legion_accessor_array_2d_write_point_c
    
        ! @see Legion::UnsafeFieldAccessor::ptr
        subroutine legion_accessor_array_3d_write_point_c(handle, point, src, bytes) &
                bind(C, name="legion_accessor_array_3d_write_point")
            use iso_c_binding
            import legion_accessor_array_3d_f_t
            import legion_point_3d_f_t
            implicit none
        
            type(legion_accessor_array_3d_f_t), value, intent(in) :: handle
            type(legion_point_3d_f_t), value, intent(in)          :: point
            type(c_ptr), value, intent(in)                        :: src
            integer(c_size_t), value, intent(in)                  :: bytes
        end subroutine legion_accessor_array_3d_write_point_c
    
        ! -----------------------------------------------------------------------
        ! File Operations
        ! -----------------------------------------------------------------------
        function legion_field_map_create_c() &
                bind(C, name="legion_field_map_create")
            use iso_c_binding
            import legion_field_map_f_t
            implicit none
        
            type(legion_field_map_f_t) :: legion_field_map_create_c
        end function legion_field_map_create_c
    
        subroutine legion_field_map_destroy_c(handle) &
                bind(C, name="legion_field_map_destroy")
            use iso_c_binding
            import legion_field_map_f_t
            implicit none
        
            type(legion_field_map_f_t), value, intent(in) :: handle
        end subroutine legion_field_map_destroy_c
    
        subroutine legion_field_map_insert_c(handle, key, value) &
                bind(C, name="legion_field_map_insert")
            use iso_c_binding
            import legion_field_map_f_t
            implicit none
        
            type(legion_field_map_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)             :: key
            type(c_ptr), value, intent(in)                :: value
        end subroutine legion_field_map_insert_c
    
        ! @see Legion::Runtime::attach_hdf5()
        function legion_runtime_attach_hdf5_c(runtime, ctx, filename, handle, parent, field_map, mode) &
                bind(C, name="legion_runtime_attach_hdf5")
            use iso_c_binding
            import legion_physical_region_f_t
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_logical_region_f_t
            import legion_field_map_f_t
            implicit none
        
            type(legion_physical_region_f_t)                   :: legion_runtime_attach_hdf5_c
            type(legion_runtime_f_t), value, intent(in)        :: runtime
            type(legion_context_f_t), value, intent(in)        :: ctx
            type(c_ptr), value, intent(in)                     :: filename
            type(legion_logical_region_f_t), value, intent(in) :: handle
            type(legion_logical_region_f_t), value, intent(in) :: parent
            type(legion_field_map_f_t), value, intent(in)      :: field_map
            integer(c_int), value, intent(in)                  :: mode
        end function legion_runtime_attach_hdf5_c
    
        ! @see Legion::Runtime::detach_hdf5()
        subroutine legion_runtime_detach_hdf5_c(runtime, ctx, region) &
                bind(C, name="legion_runtime_detach_hdf5")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_physical_region_f_t
            implicit none
        
            type(legion_runtime_f_t), value, intent(in)         :: runtime
            type(legion_context_f_t), value, intent(in)         :: ctx
            type(legion_physical_region_f_t), value, intent(in) :: region
        end subroutine legion_runtime_detach_hdf5_c
        ! -----------------------------------------------------------------------
        ! Copy Operations
        ! -----------------------------------------------------------------------
        ! @see Legion::CopyLauncher::CopyLauncher()
        function legion_copy_launcher_create_c(pred, id, launcher_tag) &
                bind(C, name="legion_copy_launcher_create")
            use iso_c_binding
            import legion_copy_launcher_f_t
            import legion_predicate_f_t
            implicit none
        
            type(legion_copy_launcher_f_t) :: legion_copy_launcher_create_c
            type(legion_predicate_f_t), value, intent(in) :: pred
            integer(c_int), value, intent(in)             :: id
            integer(c_long), value, intent(in)            :: launcher_tag
        end function legion_copy_launcher_create_c
    
        ! @see Legion::CopyLauncher::~CopyLauncher()
        subroutine legion_copy_launcher_destroy_c(handle) &
                bind(C, name="legion_copy_launcher_destroy")
            use iso_c_binding
            import legion_copy_launcher_f_t
            implicit none
        
            type(legion_copy_launcher_f_t), value, intent(in) :: handle
        end subroutine legion_copy_launcher_destroy_c
    
        ! @see Legion::Runtime::issue_copy_operation()
        subroutine legion_copy_launcher_execute_c(runtime, ctx, launcher) &
                bind(C, name="legion_copy_launcher_execute")
            use iso_c_binding
            import legion_runtime_f_t
            import legion_context_f_t
            import legion_copy_launcher_f_t
            implicit none
        
            type(legion_runtime_f_t), value, intent(in)       :: runtime
            type(legion_context_f_t), value, intent(in)       :: ctx
            type(legion_copy_launcher_f_t), value, intent(in) :: launcher
        end subroutine legion_copy_launcher_execute_c
    
        ! @see Legion::CopyLauncher::add_copy_requirements()
        function legion_copy_launcher_add_src_region_requirement_lr_c(launcher, handle, priv, prop, &
                parent, tag, verified) &
                bind(C, name="legion_copy_launcher_add_src_region_requirement_logical_region")
            use iso_c_binding
            import legion_copy_launcher_f_t
            import legion_logical_region_f_t
            implicit none
        
            integer(c_int) :: legion_copy_launcher_add_src_region_requirement_lr_c
            type(legion_copy_launcher_f_t), value, intent(in)  :: launcher
            type(legion_logical_region_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)                  :: priv
            integer(c_int), value, intent(in)                  :: prop
            type(legion_logical_region_f_t), value, intent(in) :: parent
            integer(c_long), value, intent(in)                 :: tag
            logical(c_bool), value, intent(in)                 :: verified
        end function legion_copy_launcher_add_src_region_requirement_lr_c
    
        ! @see Legion::CopyLauncher::add_copy_requirements()
        function legion_copy_launcher_add_dst_region_requirement_lr_c(launcher, handle, priv, prop, &
                parent, tag, verified) &
                bind(C, name="legion_copy_launcher_add_dst_region_requirement_logical_region")
            use iso_c_binding
            import legion_copy_launcher_f_t
            import legion_logical_region_f_t
            implicit none
        
            integer(c_int) :: legion_copy_launcher_add_dst_region_requirement_lr_c
            type(legion_copy_launcher_f_t), value, intent(in)  :: launcher
            type(legion_logical_region_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)                  :: priv
            integer(c_int), value, intent(in)                  :: prop
            type(legion_logical_region_f_t), value, intent(in) :: parent
            integer(c_long), value, intent(in)                 :: tag
            logical(c_bool), value, intent(in)                 :: verified
        end function legion_copy_launcher_add_dst_region_requirement_lr_c
    
        ! @see Legion::CopyLauncher::add_src_field()
        subroutine legion_copy_launcher_add_src_field_c(launcher, idx, fid, inst) &
                bind(C, name="legion_copy_launcher_add_src_field")
            use iso_c_binding
            import legion_copy_launcher_f_t
            implicit none
        
            type(legion_copy_launcher_f_t), value, intent(in) :: launcher
            integer(c_int), value, intent(in)                 :: idx
            integer(c_int), value, intent(in)                 :: fid
            logical(c_bool), value, intent(in)                :: inst
        end subroutine legion_copy_launcher_add_src_field_c
    
        ! @see Legion::CopyLauncher::add_dst_field()
        subroutine legion_copy_launcher_add_dst_field_c(launcher, idx, fid, inst) &
                bind(C, name="legion_copy_launcher_add_dst_field")
            use iso_c_binding
            import legion_copy_launcher_f_t
            implicit none
        
            type(legion_copy_launcher_f_t), value, intent(in) :: launcher
            integer(c_int), value, intent(in)                 :: idx
            integer(c_int), value, intent(in)                 :: fid
            logical(c_bool), value, intent(in)                :: inst
        end subroutine legion_copy_launcher_add_dst_field_c
    
        ! -----------------------------------------------------------------------
        ! Combined Operations
        ! -----------------------------------------------------------------------
        function legion_task_get_index_space_from_logical_region_c(handle, tid) &
            bind (C, name="legion_task_get_index_space_from_logical_region")
            use iso_c_binding
            import legion_index_space_f_t
            import legion_task_f_t
            implicit none
        
            type(legion_index_space_f_t)             :: legion_task_get_index_space_from_logical_region_c
            type(legion_task_f_t), value, intent(in) :: handle
            integer(c_int), value, intent(in)        :: tid
        end function legion_task_get_index_space_from_logical_region_c
        
        subroutine legion_convert_1d_to_2d_column_major_c(src, dst, offset, num_columns) &
            bind (C, name="legion_convert_1d_to_2d_column_major")
            use iso_c_binding
            import legion_byte_offset_f_t
            implicit none
            
            type(c_ptr), value, intent(in)                  :: src
            type(c_ptr), intent(in)                         :: dst(num_columns) ! this is OUT, set IN to cheat compiler
            type(legion_byte_offset_f_t), value, intent(in) :: offset
            integer(c_int), value, intent(in)               :: num_columns
        end subroutine
    end interface
end module