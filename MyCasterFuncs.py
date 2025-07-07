import numpy as np
import numpy.linalg as lg
import data_examples as dex

#Converts scatter data to image data pixel data.
def scatter_to_im(scatter_data,res):
  """args: scatter_data,res ;  takes in raw scatter x y coordinate scatter
  data. Put in your data(numpy array of a list of coordinates), and then decide the
  resolution of the image you want."""
  data = scatter_data
  # # data = 10*rng.normal(size = (100,2))
  # plt.scatter(data[:,0] , data[:,1])
  # plt.show()

  data_norm = (data - data.min())/(data.max() - data.min())
  # plt.scatter(data_norm[:,0] , data_norm[:,1])
  # plt.show()
  # res = 100

  d = np.floor((data_norm*(res-1))).astype(int)

  I = np.zeros((res,res))

  for p in d:

    I[(res- 1) - p[1] , p[0]] = 1




#   plt.imshow(I)
#   plt.show()

  return I





#######Image converter with padder
# def scatter_to_im_w_pad(scatter_data,res,pad):
#   """args: scatter_data,res ;  put in your data, and then decide the
#   resolution of the image you want."""

#   data = scatter_data
#   # # data = 10*rng.normal(size = (100,2))
#   # plt.scatter(data[:,0] , data[:,1])
#   # plt.show()

#   data_norm = (data - data.min())/(data.max() - data.min())
#   # plt.scatter(data_norm[:,0] , data_norm[:,1])
#   # plt.show()
#   # res = 100

#   d = np.floor((data_norm*(res-1))).astype(int)

#   d = d + np.array([pad,pad])

#   I = np.zeros((res + 2*pad,res+2*pad))

#   for p in d:

#     I[ 2*pad + res - p[1] ,p[0]] = 1




# #   plt.imshow(I)
# #   plt.show()

#   return I







##################Convolution function

# def convolve(kernel , I):

#   pad = len(kernel)

#   A = np.zeros((len(I) - pad,len(I) - pad))



#   for i in range(len(I) - pad ):
#     for j in range(len(I) - pad):

#       A[i, j ] = (kernel * I[i: i + pad , j : j + pad]).sum()

#   return A


def scatter_to_im_w_pad(scatter_data,res,pad,point_list = False,sizes = None,bounding_dist = None):
  """args: scatter_data,res ;  put in your data, and then decide the
  resolution of the image you want."""

  # # data = 10*rng.normal(size = (100,2))
  # plt.scatter(data[:,0] , data[:,1])
  # plt.show()
  data = scatter_data

  #l_inf dist 

#   bounding_dist = 100

  data_norm = (data  + bounding_dist) /(2*bounding_dist)




#   data_norm = (data - data.min())/(data.max() - data.min())
  # plt.scatter(data_norm[:,0] , data_norm[:,1])
  # plt.show()
  # res = 100

  d = np.floor((data_norm*(res-1))).astype(int)

  d = d + np.array([pad,pad])

  I = np.zeros((res + 2*pad,res+ 2*pad))



  if point_list == True:
    # kernel = np.ones((3,3))

    #for development
    # sizes = rng.integers(1,3,size = len(d))

    A = np.zeros(I.shape)


##############ORIGINALLY 7
    size_par = 7
    # size_array = np.round(((sizes - sizes.min())/(sizes.max() - \
    size_array = np.round(((sizes)/(sizes.max()) * size_par)).astype(int)


    for u ,p in enumerate(d):

        if p[0] < pad or p[0]>= res + pad or p[1] < pad or p[1]>= res + pad:
            continue

        # if False:
        #     continue

        else:

            #   p_size = int(np.round(sizes[u]/min_size))

            #standard thickening
            kernel = np.ones((2*size_array[u] + 1,2*size_array[u] + 1))


            ######
            # xx , yy = np.mgrid[:2*sizes[u] + 1 , :2*sizes[u] + 1 ]
            # xx = xx - (( sizes[u] - 1)/2 + 1)
            # yy = yy - (( sizes[u] - 1)/2 + 1)

            # kernel = st.norm.pdf(xx)*st.norm.pdf(yy)*sizes[u]*10**6
            
            
            ###For reciprical norm kernel
            # kernel = 1/(xx**2 + yy**2 + 0.25)* sizes[u] * 100


            # print(kernel.max())
            # print(st.norm.pdf(5))
      


      


            kernel_half_len = int((len(kernel) - 1)/2 )
            # print(kernel_half_len)
            #####Maybe
            # kernel_half_len = kernel_half_len
            ####







            p_loc = np.array([2*pad + res - p[1] ,p[0]]).astype(int)

            #change 'kerenel to 1 to get solid blocks.
            # I[p_loc[0] - kernel_half_len:p_loc[0] + kernel_half_len + 1,p_loc[1] - kernel_half_len:p_loc[1] + kernel_half_len + 1] = kernel
            
            # A[p_loc[0] - kernel_half_len:p_loc[0] + kernel_half_len + 1,p_loc[1] - kernel_half_len:p_loc[1] + kernel_half_len + 1] += kernel

            A[p_loc[0] - kernel_half_len:p_loc[0] + kernel_half_len + 1,p_loc[1] - kernel_half_len:p_loc[1] + kernel_half_len + 1] = 1






    I = A
    # plt.imshow(I)
    # plt.show()




    return I




  for p in d:
    if False:
        continue
    # if p[0] < 0 or p[0]>= res or p[1] < 0 or p[1]>= res:
    #   continue
    else:

      I[ 2*pad + res - p[1] ,p[0]] = 1






#   plt.imshow(I,vmin = 0,vmax = 1)
#   plt.show()

  return I













  ###############Caster Function

def cast_data_with_up_rotation(vp,c,data = dex.nested_cube_data(),scale_factor_red_line = None):
  

#Test Data #########
#   if data == None:
#     xx,yy,zz = np.indices((21,21,21))
#     data = np.stack([xx.ravel() , yy.ravel() , zz.ravel()]).T

#     data = data.astype(float)



#     data = 2*(data/ 20 - 0.5)



#     mask = (np.sort(np.abs(data ), axis = 1)== 1)[:,-2:].all(axis = 1)



#     data = data[mask].astype(float)

#     data = np.concatenate([data,data / 2])

    #Red stuff
    # scale_factor_red_line = 0.5-0.05 #1 for touching corner.
    # data = np.concatenate([data,np.zeros((1,3))])
    # data = np.concatenate([data,  scale_factor_red_line* np.ones((1,3))])

  #################end test data

  #Unit Vectors
  i = np.array([1,0,0])
  j = np.array([0,1,0])
  k = np.array([0,0,1])

  #Calculating viewing plane 2d coordinate system


  
  c_pi = c[0]*i + c[1]*j
  c_pi_norm = lg.norm(c_pi)
  c_k_rel = -c[2]*c_pi/c_pi_norm + c_pi_norm*k
  k_rel = c_k_rel/lg.norm(c_k_rel)

  j_rel = np.array([-c[1] , c[0] , 0])/lg.norm(np.array([-c[1] , c[0] , 0]))



  #Calculating the rays from our vanishing point to each data point in point cloud
  rays = (data - vp)
  data_proj = vp +  rays*lg.norm(c - vp)**2  /  ((rays)@(c - vp).T )[:,None]


 
  #The value of each data point in our 2d viewing plane coordinate system
  y_rel_vals = (data_proj - c )@j_rel
  z_rel_vals = (data_proj - c )@k_rel

 

  #Calculating the size of each point in our 2d image
  sizes = lg.norm(data_proj - vp,axis = 1)**2/lg.norm(data - vp,axis = 1)**2


  #for ploting the window
  # plt.scatter(y_rel_vals,data_z_rel ,s = 10*sizes)
  # plt.xlim(-3,3)
  # plt.ylim(-3,3)
  # plt.show()

  #returns scatter data
  return y_rel_vals, z_rel_vals, 10*sizes,j_rel,k_rel


##############Test to see if matplotlib is working.


# import numpy as np
# import matplotlib.pyplot as plt

# rng = np.random.default_rng()

# data = 10*rng.normal(size = (100,2))
# # plt.scatter(data[:,0] , data[:,1])
# # plt.show()

# data_norm = (data - data.min())/(data.max() - data.min())
# # plt.scatter(data_norm[:,0] , data_norm[:,1])
# # plt.show()




# I = scatter_to_im(scatter_data = data,res = 100)
# plt.imshow(I)
# plt.show()


i = np.array([1,0,0])
j = np.array([0,1,0])
k = np.array([0,0,1])

def my_path_coord(theta):

  k = np.array([0,0,1])

  cos = np.cos(theta)
  sin = np.sin(theta)

  v_theta = np.array([cos,sin,0])
  p = v_theta + (cos/2 + 0.5)*v_theta + sin*k



  return p