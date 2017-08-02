module legion_fortran_object_oriented
    use, intrinsic :: iso_c_binding
    use legion_fortran_types
    use legion_fortran_c_interface
    implicit none
    
    ! Point Class
    type LegionPoint
        integer :: dim
    end type LegionPoint

    type, extends(LegionPoint) :: LegionPoint1D
        type(legion_point_1d_f_t) :: point
    end type LegionPoint1D
    
    type, extends(LegionPoint) :: LegionPoint2D
        type(legion_point_2d_f_t) :: point
    end type LegionPoint2D
    
    type, extends(LegionPoint) :: LegionPoint3D
        type(legion_point_3d_f_t) :: point
    end type LegionPoint3D
    
    interface LegionPoint1D
        module procedure legion_point_1d_constructor_integer4
        module procedure legion_point_1d_constructor_integer8
    end interface
    
    interface LegionPoint2D
        module procedure legion_point_2d_constructor_integer4
        module procedure legion_point_2d_constructor_integer8
    end interface
    
    interface LegionPoint3D
        module procedure legion_point_3d_constructor_integer4
        module procedure legion_point_3d_constructor_integer8
    end interface
    
    ! Accessor Class
    type LegionFieldAccessor
        integer :: dim
        integer(c_size_t) :: data_size
    contains
        procedure :: init => legion_field_accessor_init
        procedure, private :: legion_field_accessor_read_point_ptr
        procedure, private :: legion_field_accessor_read_point_integer4
        procedure, private :: legion_field_accessor_read_point_integer8
        procedure, private :: legion_field_accessor_read_point_real4
        procedure, private :: legion_field_accessor_read_point_real8
        procedure, private :: legion_field_accessor_write_point_ptr
        procedure, private :: legion_field_accessor_write_point_integer4
        procedure, private :: legion_field_accessor_write_point_integer8
        procedure, private :: legion_field_accessor_write_point_real4
        procedure, private :: legion_field_accessor_write_point_real8
        generic :: read_point => legion_field_accessor_read_point_ptr, &
                                 legion_field_accessor_read_point_integer4, legion_field_accessor_read_point_integer8, &
                                 legion_field_accessor_read_point_real4, legion_field_accessor_read_point_real8
        generic :: write_point => legion_field_accessor_write_point_ptr, &
                                  legion_field_accessor_write_point_integer4, legion_field_accessor_write_point_integer8, &
                                  legion_field_accessor_write_point_real4, legion_field_accessor_write_point_real8
    end type LegionFieldAccessor

    type, extends(LegionFieldAccessor) :: LegionFieldAccessor1D
        type(legion_accessor_array_1d_f_t) :: accessor
    end type LegionFieldAccessor1D
    
    type, extends(LegionFieldAccessor) :: LegionFieldAccessor2D
        type(legion_accessor_array_2d_f_t) :: accessor
    end type LegionFieldAccessor2D
    
    type, extends(LegionFieldAccessor) :: LegionFieldAccessor3D
        type(legion_accessor_array_3d_f_t) :: accessor
    end type LegionFieldAccessor3D
    
    interface LegionFieldAccessor1D
        module procedure legion_field_accessor_1d_constructor
    end interface
    
    interface LegionFieldAccessor2D
        module procedure legion_field_accessor_2d_constructor
    end interface
    
    interface LegionFieldAccessor3D
        module procedure legion_field_accessor_3d_constructor
    end interface
    

contains
    
    function legion_point_1d_constructor_integer4(x)
        implicit none
        
        type(LegionPoint1D)         :: legion_point_1d_constructor_integer4
        integer(kind=4), intent(in) :: x
        
        legion_point_1d_constructor_integer4%point%x(0) = int(x, 8)
        
    end function legion_point_1d_constructor_integer4
    
    function legion_point_1d_constructor_integer8(x)
        implicit none
        
        type(LegionPoint1D)         :: legion_point_1d_constructor_integer8
        integer(kind=8), intent(in) :: x
        
        legion_point_1d_constructor_integer8%point%x(0) = x
        
    end function legion_point_1d_constructor_integer8
    
    function legion_point_2d_constructor_integer4(x, y)
        implicit none
        
        type(LegionPoint2D)         :: legion_point_2d_constructor_integer4
        integer(kind=4), intent(in) :: x
        integer(kind=4), intent(in) :: y
        
        legion_point_2d_constructor_integer4%point%x(0) = int(x, 8)
        legion_point_2d_constructor_integer4%point%x(1) = int(y, 8)
        
    end function legion_point_2d_constructor_integer4
    
    function legion_point_2d_constructor_integer8(x, y)
        implicit none
        
        type(LegionPoint2D)         :: legion_point_2d_constructor_integer8
        integer(kind=8), intent(in) :: x
        integer(kind=8), intent(in) :: y
        
        legion_point_2d_constructor_integer8%point%x(0) = x
        legion_point_2d_constructor_integer8%point%x(1) = y
        
    end function legion_point_2d_constructor_integer8
    
    function legion_point_3d_constructor_integer4(x, y, z)
        implicit none
        
        type(LegionPoint3D)         :: legion_point_3d_constructor_integer4
        integer(kind=4), intent(in) :: x
        integer(kind=4), intent(in) :: y
        integer(kind=4), intent(in) :: z
        
        legion_point_3d_constructor_integer4%point%x(0) = int(x, 8)
        legion_point_3d_constructor_integer4%point%x(1) = int(y, 8)
        legion_point_3d_constructor_integer4%point%x(2) = int(z, 8)
        
    end function legion_point_3d_constructor_integer4
    
    function legion_point_3d_constructor_integer8(x, y, z)
        implicit none
        
        type(LegionPoint3D)         :: legion_point_3d_constructor_integer8
        integer(kind=8), intent(in) :: x
        integer(kind=8), intent(in) :: y
        integer(kind=8), intent(in) :: z
        
        legion_point_3d_constructor_integer8%point%x(0) = x
        legion_point_3d_constructor_integer8%point%x(1) = y
        legion_point_3d_constructor_integer8%point%x(2) = z
        
    end function legion_point_3d_constructor_integer8

    function legion_field_accessor_1d_constructor(physical_region, fid, data_size)
        implicit none
        
        type(LegionFieldAccessor1D)                  :: legion_field_accessor_1d_constructor
        type(legion_physical_region_f_t), intent(in) :: physical_region
        integer(c_int), intent(in)                   :: fid
        integer(c_size_t), intent(in)                :: data_size
        

        legion_field_accessor_1d_constructor%dim = 1
        legion_field_accessor_1d_constructor%data_size = data_size
        legion_field_accessor_1d_constructor%accessor = legion_physical_region_get_field_accessor_array_1d_c(physical_region, fid)
    end function legion_field_accessor_1d_constructor
    
    function legion_field_accessor_2d_constructor(physical_region, fid, data_size)
        implicit none
        
        type(LegionFieldAccessor2D)                  :: legion_field_accessor_2d_constructor
        type(legion_physical_region_f_t), intent(in) :: physical_region
        integer(c_int), intent(in)                   :: fid
        integer(c_size_t), intent(in)                :: data_size
        

        legion_field_accessor_2d_constructor%dim = 1
        legion_field_accessor_2d_constructor%data_size = data_size
        legion_field_accessor_2d_constructor%accessor = legion_physical_region_get_field_accessor_array_2d_c(physical_region, fid)
    end function legion_field_accessor_2d_constructor
    
    function legion_field_accessor_3d_constructor(physical_region, fid, data_size)
        implicit none
        
        type(LegionFieldAccessor3D)                  :: legion_field_accessor_3d_constructor
        type(legion_physical_region_f_t), intent(in) :: physical_region
        integer(c_int), intent(in)                   :: fid
        integer(c_size_t), intent(in)                :: data_size
        

        legion_field_accessor_3d_constructor%dim = 1
        legion_field_accessor_3d_constructor%data_size = data_size
        legion_field_accessor_3d_constructor%accessor = legion_physical_region_get_field_accessor_array_3d_c(physical_region, fid)
    end function legion_field_accessor_3d_constructor
        
    subroutine legion_field_accessor_init(this, physical_region, fid, data_size)
        implicit none
        
        class(LegionFieldAccessor), intent(inout)    :: this
        type(legion_physical_region_f_t), intent(in) :: physical_region
        integer(c_int), intent(in)                   :: fid
        integer(c_size_t), intent(in)                  :: data_size
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            this%dim = 1
            this%data_size = data_size
            this%accessor = legion_physical_region_get_field_accessor_array_1d_c(physical_region, fid)
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_init
    
    subroutine legion_field_accessor_read_point_ptr(this, point, dst)
        implicit none
        
        class(LegionFieldAccessor), intent(in) :: this
        class(LegionPoint), intent(in)         :: point
        type(c_ptr), intent(out)               :: dst 
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            select type (point)
            type is (LegionPoint1D)
                call legion_accessor_array_1d_read_point_c(this%accessor, point%point, dst, this%data_size)
            end select
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_read_point_ptr
    
    subroutine legion_field_accessor_read_point_integer4(this, point, dst)
        implicit none
        
        class(LegionFieldAccessor), intent(in) :: this
        class(LegionPoint), intent(in)         :: point
        integer(kind=4), target, intent(out)   :: dst
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            select type (point)
            type is (LegionPoint1D)
                call legion_accessor_array_1d_read_point_c(this%accessor, point%point, c_loc(dst), this%data_size)
            end select
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_read_point_integer4
    
    subroutine legion_field_accessor_read_point_integer8(this, point, dst)
        implicit none
        
        class(LegionFieldAccessor), intent(in) :: this
        class(LegionPoint), intent(in)         :: point
        integer(kind=8), target, intent(out)   :: dst
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            select type (point)
            type is (LegionPoint1D)
                call legion_accessor_array_1d_read_point_c(this%accessor, point%point, c_loc(dst), this%data_size)
            end select
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_read_point_integer8
    
    subroutine legion_field_accessor_read_point_real4(this, point, dst)
        implicit none
        
        class(LegionFieldAccessor), intent(in) :: this
        class(LegionPoint), intent(in)         :: point
        real(kind=4), target, intent(out)      :: dst
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            select type (point)
            type is (LegionPoint1D)
                call legion_accessor_array_1d_read_point_c(this%accessor, point%point, c_loc(dst), this%data_size)
            end select
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_read_point_real4
    
    subroutine legion_field_accessor_read_point_real8(this, point, dst)
        implicit none
        
        class(LegionFieldAccessor), intent(in) :: this
        class(LegionPoint), intent(in)         :: point
        real(kind=8), target, intent(out)      :: dst
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            select type (point)
            type is (LegionPoint1D)
                call legion_accessor_array_1d_read_point_c(this%accessor, point%point, c_loc(dst), this%data_size)
            end select
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_read_point_real8
      
    subroutine legion_field_accessor_write_point_ptr(this, point, src)
        implicit none
    
        class(LegionFieldAccessor), intent(in) :: this
        class(LegionPoint), intent(in)         :: point
        type(c_ptr), intent(in)                :: src
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            select type (point)
            type is (LegionPoint1D)
                call legion_accessor_array_1d_write_point_c(this%accessor, point%point, src, this%data_size)
            end select
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_write_point_ptr
    
    subroutine legion_field_accessor_write_point_integer4(this, point, src)
        implicit none
    
        class(LegionFieldAccessor), intent(in) :: this
        class(LegionPoint), intent(in)         :: point
        integer(kind=4), target, intent(in)    :: src
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            select type (point)
            type is (LegionPoint1D)
                call legion_accessor_array_1d_write_point_c(this%accessor, point%point, c_loc(src), this%data_size)
            end select
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_write_point_integer4
    
    subroutine legion_field_accessor_write_point_integer8(this, point, src)
        implicit none
    
        class(LegionFieldAccessor), intent(in) :: this
        class(LegionPoint), intent(in)         :: point
        integer(kind=8), target, intent(in)    :: src
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            select type (point)
            type is (LegionPoint1D)
                call legion_accessor_array_1d_write_point_c(this%accessor, point%point, c_loc(src), this%data_size)
            end select
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_write_point_integer8
    
    subroutine legion_field_accessor_write_point_real4(this, point, src)
        implicit none
    
        class(LegionFieldAccessor), intent(in) :: this
        class(LegionPoint), intent(in)         :: point
        real(kind=4), target, intent(in)       :: src
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            select type (point)
            type is (LegionPoint1D)
                call legion_accessor_array_1d_write_point_c(this%accessor, point%point, c_loc(src), this%data_size)
            end select
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_write_point_real4
    
    subroutine legion_field_accessor_write_point_real8(this, point, src)
        implicit none
    
        class(LegionFieldAccessor), intent(in) :: this
        class(LegionPoint), intent(in)         :: point
        real(kind=8), target, intent(in)       :: src
        
        select type (this)
        type is (LegionFieldAccessor)
              ! no further initialization required
        class is (LegionFieldAccessor1D)
            ! 1D
            select type (point)
            type is (LegionPoint1D)
                call legion_accessor_array_1d_write_point_c(this%accessor, point%point, c_loc(src), this%data_size)
            end select
        class default
          ! give error for unexpected/unsupported type
             stop 'initialize: unexpected type for LegionFieldAccessor object!'
        end select
    end subroutine legion_field_accessor_write_point_real8

end module