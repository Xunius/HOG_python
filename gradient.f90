module GRADIENT
implicit none

private
integer, parameter :: ikind = selected_real_kind(p=15,r=10)

public COMPUTE_GRAD3, COMPUTE_GRAD2

contains

    !-----------Compute y- and x- gradients-----------
    subroutine COMPUTE_GRAD3(x,ny,nx,gy,gx)
    ! Compute y- and x- gradients
        implicit none

        !integer, parameter :: ikind = selected_real_kind(p=15,r=10)

        integer, intent(in) :: ny,nx
        real(kind=8), dimension(ny,nx,3), intent(in) :: x
        real(kind=8), dimension(ny,nx,3), intent(out) :: gy,gx

        gy=0.5d0*(cshift(x,1,1)-cshift(x,-1,1))
        gx=0.5d0*(cshift(x,1,2)-cshift(x,-1,2))

        !------------------Correct edges------------------
        gy(1,:,:)=x(2,:,:)-x(1,:,:)
        gy(ny,:,:)=x(ny,:,:)-x(ny-1,:,:)
        gx(:,1,:)=x(:,2,:)-x(:,1,:)
        gx(:,nx,:)=x(:,nx,:)-x(:,nx-1,:)

    end subroutine COMPUTE_GRAD3


    subroutine COMPUTE_GRAD2(x,ny,nx,gy,gx)
    ! Compute y- and x- gradients
        implicit none

        integer, parameter :: ikind = selected_real_kind(p=15,r=10)

        integer, intent(in) :: ny,nx
        real(kind=ikind), dimension(ny,nx), intent(in) :: x
        real(kind=ikind), dimension(ny,nx), intent(out) :: gy,gx

        gy=0.5_ikind*(cshift(x,1,1)-cshift(x,-1,1))
        gx=0.5_ikind*(cshift(x,1,2)-cshift(x,-1,2))

        !------------------Correct edges------------------
        gy(1,:)=x(2,:)-x(1,:)
        gy(ny,:)=x(ny,:)-x(ny-1,:)
        gx(:,1)=x(:,2)-x(:,1)
        gx(:,nx)=x(:,nx)-x(:,nx-1)

    end subroutine COMPUTE_GRAD2

end module GRADIENT
