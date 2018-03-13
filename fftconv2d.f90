module FFTCONV2D
implicit none

private
public :: CONV2D_FULL, CONV2D_SAME, CONV2D_VALID

contains

    !---------Compute 2D convolution using FFT, mode full---------
    subroutine CONV2D_FULL(slab,resslab,kernel,ny,nx,ky,kx)
    ! Compute 2D convolution using FFT, mode full

    ! <slab>: in, real, 2D array, data to convolve.
    ! <kernel>: in, real, 2D array, kernel to convolve with.
    ! <ny>, <nx>: in, int, shape of <slab>.
    ! <ky>, <kx>: in, int, shape of <kernel>.

    ! Return <resslab>: out, real, 2D array, convolution result.
    !
    ! 2D convolution is obtained using the convolution theorem:
    !
    !    conv(f, g) = F^{-1}{ F{f} * F{g} }
    !
    ! where F{} is the Fourier transformation, F^{-1} the inverse transformation.
    ! * is the point-wise multiplication
    !
    ! To make the convolution non-circular, data in each dimension is padded
    ! to n+k-1 (n being data size, k being kernel size), before padding to
    ! the next power of 2.
    !
    ! To achieve max speed boost, data are padded with zeros to the next
    ! power of 2 before doing the FFT and iFFT
    !

        implicit none
        include "fftw3.f"

        integer, parameter :: ikind = selected_real_kind(p=15,r=10)

        integer, intent(in) :: ny, nx, ky, kx
        real(kind=ikind), dimension(ny,nx), intent(in) :: slab
        real(kind=ikind), dimension(ky,kx), intent(in) :: kernel
        real(kind=ikind), dimension(ny+ky-1,nx+kx-1), intent(out) :: resslab

        !----------------Intermediate vars----------------
        complex(kind=ikind), dimension(:,:), allocatable :: fc1, fc2, fcprod
        real(kind=ikind), dimension(:,:), allocatable :: slab_pad, kernel_pad, res_pad
        integer(kind=ikind) :: plan_forward, plan_backward
        integer :: nh, ny2, nx2, pad_y, pad_x

        !------------------Compute shapes-----------------
        ! make convolution from circular to linear
        ny2=ny+ky-1
        nx2=nx+kx-1
        ! get the next power of 2
        pad_y=2**(ceiling(log(float(ny2))/log(2.)))
        pad_x=2**(ceiling(log(float(nx2))/log(2.)))

        !---------------------Pad slab---------------------
        allocate(slab_pad(pad_y,pad_x))
        slab_pad=0._ikind
        slab_pad(1:ny,1:nx)=slab

        !--------------------Pad kernel--------------------
        allocate(kernel_pad(pad_y,pad_x))
        kernel_pad=0._ikind
        kernel_pad(1:ky, 1:kx) = kernel(ky:1:-1, kx:1:-1) ! flip kernel

        !--------------FFT on slab and kernel--------------
        nh = pad_y/2+1
        allocate(fc1(nh,pad_x))
        allocate(fc2(nh,pad_x))
        allocate(fcprod(nh,pad_x))
        allocate(res_pad(pad_y,pad_x))

        ! Compute FFT on slab
        call dfftw_plan_dft_r2c_2d_(plan_forward, pad_y, pad_x, slab_pad, fc1, FFTW_ESTIMATE)
        call dfftw_execute_dft_r2c(plan_forward,slab_pad,fc1)
        call dfftw_destroy_plan_(plan_forward)

        ! Compute FFT on kernel
        call dfftw_plan_dft_r2c_2d_(plan_forward, pad_y, pad_x, kernel_pad, fc2, FFTW_ESTIMATE)
        call dfftw_execute_dft_r2c(plan_forward,kernel_pad,fc2)
        call dfftw_destroy_plan_(plan_forward)

        !-----------------iFFT on product-----------------
        fcprod=fc1*fc2
        call dfftw_plan_dft_c2r_2d_(plan_backward, pad_y, pad_x, fcprod, res_pad, FFTW_ESTIMATE)
        call dfftw_execute_dft_c2r(plan_backward,fcprod,res_pad)
        call dfftw_destroy_plan_(plan_backward)

        res_pad=res_pad/pad_x/pad_y
        ! crop result
        resslab=res_pad(1:ny2, 1:nx2)

        deallocate(fc1)
        deallocate(fc2)
        deallocate(fcprod)
        deallocate(res_pad)
        
    end subroutine CONV2D_FULL


    !---------Compute 2D convolution using FFT, mode same---------
    subroutine CONV2D_SAME(slab,resslab,kernel,ny,nx,ky,kx)
    ! Compute 2D convolution using FFT, mode same

    ! <slab>: in, real, 2D array, data to convolve.
    ! <kernel>: in, real, 2D array, kernel to convolve with.
    ! <ny>, <nx>: in, int, shape of <slab>.
    ! <ky>, <kx>: in, int, shape of <kernel>.

    ! Return <resslab>: out, real, 2D array, convolution result.

        implicit none

        integer, parameter :: ikind = selected_real_kind(p=15,r=10)

        integer, intent(in) :: ny, nx, ky, kx
        real(kind=ikind), dimension(ny,nx), intent(in) :: slab
        real(kind=ikind), dimension(ky,kx), intent(in) :: kernel
        real(kind=ikind), dimension(ny,nx), intent(out) :: resslab

        !----------------Intermediate vars----------------
        real(kind=ikind), dimension(ny+ky-1,nx+kx-1) :: conv_full
        integer :: start_y, start_x, end_y, end_x

        !-------------Do full mode convolution-------------
        call CONV2D_FULL(slab,conv_full,kernel,ny,nx,ky,kx)

        !------------------Crop same area------------------
        start_y=(ky-1)/2+1
        start_x=(kx-1)/2+1
        end_y=start_y+ny
        end_x=start_x+nx
        resslab=conv_full(start_y : end_y, start_x : end_x)
        
    end subroutine CONV2D_SAME



    !---------Compute 2D convolution using FFT, mode valid---------
    subroutine CONV2D_VALID(slab,resslab,kernel,ny,nx,ky,kx)
    ! Compute 2D convolution using FFT, mode valid

    ! <slab>: in, real, 2D array, data to convolve.
    ! <kernel>: in, real, 2D array, kernel to convolve with.
    ! <ny>, <nx>: in, int, shape of <slab>.
    ! <ky>, <kx>: in, int, shape of <kernel>.

    ! Return <resslab>: out, real, 2D array, convolution result.

        implicit none

        integer, parameter :: ikind = selected_real_kind(p=15,r=10)

        integer, intent(in) :: ny, nx, ky, kx
        real(kind=ikind), dimension(ny,nx), intent(in) :: slab
        real(kind=ikind), dimension(ky,kx), intent(in) :: kernel
        real(kind=ikind), dimension(ny-ky+1,nx-kx+1), intent(out) :: resslab

        !----------------Intermediate vars----------------
        real(kind=ikind), dimension(ny+ky-1,nx+kx-1) :: conv_full

        !-------------Do full mode convolution-------------
        call CONV2D_FULL(slab,conv_full,kernel,ny,nx,ky,kx)

        !------------------Crop same area------------------
        resslab=conv_full(ky : ny+1, kx : nx+1)
        
    end subroutine CONV2D_VALID

end module FFTCONV2D
