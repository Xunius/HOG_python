module HOG
use gradient
use fftconv2d
implicit none

private
public COMPUTE_HOG

contains

    !-Compute HOG feature descriptors from input image-
    subroutine COMPUTE_HOG(img,height,width,ws_y,ws_x,stride,orientations, &
        & ppc_y,ppc_x,cpb_y,cpb_x, &
        & block_norm, transform_sqrt, feature_vector, feature, convs)
    ! Compute HOG feature descriptors from input image
        implicit none

        integer, intent(in) :: height,width,ws_y,ws_x,ppc_y,ppc_x,cpb_y,cpb_x
        integer, intent(in) :: stride,orientations
        integer, intent(in) :: block_norm, transform_sqrt, feature_vector
        integer(kind=8), dimension(height,width,3), intent(in) :: img

        real(kind=8), dimension(((height-ws_y)/stride+1)*((width-ws_x)/stride+1), &
            & ws_y/ppc_y-cpb_y+1,ws_x/ppc_x-cpb_x+1, &
            & cpb_y,cpb_x,orientations), intent(out) :: feature

        ! intermediate vars
        real(kind=8), dimension(height,width,3) :: img_f, weights3
        real(kind=8), dimension(height,width) :: theta, weights
        integer, dimension(height,width)  :: maxidx
        real(kind=8), dimension(height,width,3) :: gy,gx,theta3
        real(kind=8), dimension(height,width) :: countii
        real(kind=8), dimension(ppc_y,ppc_x) :: kernel
        real(kind=8), dimension(height-ppc_y+1,width-ppc_x+1,orientations), intent(out) :: convs
        real(kind=8), dimension(:,:,:,:), allocatable :: hists, blockii, normii
        real(kind=8), dimension(:,:), allocatable :: convii
        real(kind=8), dimension(:), allocatable :: normii_array

        real(kind=8) :: PI
        real(kind=8) :: dtheta, tmp
        integer :: i, j, m1, m2, n1, n2, t, b1, b2
        integer, allocatable, dimension(:) :: dy0, dx0, dy, dx
        integer, dimension(orientations) :: dz0
        integer, dimension(4) :: shape_list
        real(kind=8) :: t2,t1

        call cpu_time(t1)
        PI=4.D0*DATAN(1.D0)
        img_f=dble(img)

        if (transform_sqrt == 1) then
            img_f=sqrt(img_f)
        end if

        !-----Compute gradient orientations in [0,180]-----
        call COMPUTE_GRAD3(img_f,height,width,gy,gx)

        weights3=sqrt(gy**2+gx**2)
        
        theta3=datan2(gy,gx)*180.d0/PI
        theta3=mod(theta3+180.d0,180.d0)

        theta=0.d0
        weights=maxval(weights3,3)
        maxidx=maxloc(weights3,3)

        do i = 1,3
            theta=merge(theta3(:,:,i),theta,maxidx==i)
        end do

        dtheta=180.d0/orientations

        !--------------Compute 3d convolution--------------
        kernel=1.d0

        allocate(convii(height-ppc_y+1, width-ppc_x+1))

        do i = 1,orientations
            countii=merge(1.d0, 0.d0, theta>=dtheta*(i-1) .and. theta<dtheta*i)
            countii=countii*weights
            call CONV2D_VALID(countii,convii,kernel,height,width,ppc_y,ppc_x)
            convs(:,:,i)=convii/ppc_x/ppc_y
        end do

        call cpu_time(t2)
        write(*,*) 'time',t2-t1

        m1=ws_y/ppc_y
        m2=ws_x/ppc_x
        n1=(height-ws_y)/stride+1
        n2=(width-ws_x)/stride+1

        !---------------Get strided indices---------------
        allocate(hists(n1*n2,m1,m2,orientations))
        allocate(dy0(m1))
        allocate(dx0(m2))
        allocate(dy(m1))
        allocate(dx(m2))

        dy0 = (/ (i, i=1, ppc_y*m1, ppc_y) /)
        dx0 = (/ (i, i=1, ppc_x*m2, ppc_x) /)
        dz0 = (/ (i, i=1, orientations, 1) /)

        do t = 0,n1*n2-1
            i=t/n2
            j=mod(t,n2)
            dy=dy0+stride*i
            dx=dx0+stride*j
            hists(t+1,:,:,:)=convs(dy,dx,:)
        end do

        !--------------Normalize over blocks--------------
        b1=m1-cpb_y+1
        b2=m2-cpb_x+1

        allocate(blockii(n1*n2,cpb_y,cpb_x,orientations))
        allocate(normii(n1*n2,cpb_y,cpb_x,orientations))
        allocate(normii_array(n1*n2))
        shape_list = [ n1*n2, 1, 1, 1 ] 

        do t = 0,b1*b2-1
            i=t/b2+1
            j=mod(t,b2)+1

            blockii=hists(:,i:i+cpb_y-1, j:j+cpb_x-1, :)
            if (block_norm == 1) then
                if (n1*n2==1) then
                    tmp=sum(abs(blockii))
                    blockii=blockii/(tmp+1e-5)
                else
                    normii_array=sum(sum(sum(abs(blockii),4),3),2)
                    normii=spread(spread(spread(normii_array,2,cpb_y),3,cpb_x),4,orientations)
                    blockii=blockii/(normii+1d-5)
                end if
            end if
            feature(:,i,j,:,:,:)=blockii
        end do

        deallocate(convii)
        deallocate(hists)
        deallocate(dy0)
        deallocate(dx0)
        deallocate(dy)
        deallocate(dx)
        deallocate(blockii)
        deallocate(normii)

        
    end subroutine COMPUTE_HOG

end module HOG
