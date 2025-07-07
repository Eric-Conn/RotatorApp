import numpy as np
import numpy.linalg as lg

i = np.array([1,0,0])
j = np.array([0,1,0])
k = np.array([0,0,1])

def nested_cube_data():
    xx,yy,zz = np.indices((21,21,21))
    data = np.stack([xx.ravel() , yy.ravel() , zz.ravel()]).T

    data = data.astype(float)



    data = 2*(data/ 20 - 0.5)



    mask = (np.sort(np.abs(data ), axis = 1)== 1)[:,-2:].all(axis = 1)



    data = data[mask].astype(float)

    data = np.concatenate([data,data / 2])


    i_axis = np.arange(30)[:,None]*i[None,:]*(0.25/30)
    j_axis = np.arange(30)[:,None]*j[None,:]*(0.5/30)
    k_axis = np.arange(30)[:,None]*k[None,:]*(0.75/30)

    

    data = np.concatenate([data,-i_axis,j_axis , k_axis])



    return data



def L_of_cubes():
    xx,yy,zz = np.indices((21,21,21))
    data = np.stack([xx.ravel() , yy.ravel() , zz.ravel()]).T

    data = data.astype(float)



    data = 2*(data/ 20 - 0.5)



    mask = (np.sort(np.abs(data ), axis = 1)== 1)[:,-2:].all(axis = 1)



    data = data[mask].astype(float)

    data_i = data + i

    data_j = data + j

    data_k = data + k

    data = np.concatenate([data_i,data_j,data_k])






    # data = np.concatenate([data,data / 2])




    return data


def long_block():
    xx,yy,zz = np.indices((21,21,21))
    data = np.stack([xx.ravel() , yy.ravel() , zz.ravel()]).T

    data = data.astype(float)



    data = 2*(data/ 20 - 0.5)



    mask = (np.sort(np.abs(data ), axis = 1)== 1)[:,-2:].all(axis = 1)



    data = data[mask].astype(float)


    lim = 4
    ns = np.arange(1,lim)

    for n in ns:

        data = np.concatenate([data , data + n*2*i, data - n*2*i])

    

        



    # data_i = data + i

    # data_j = data + j

    # data_k = data + k

    # data = np.concatenate([data_i,data_j,data_k])






    # data = np.concatenate([data,data / 2])




    return data



def I_beam():
    xx,yy,zz = np.indices((21,21,21))
    data = np.stack([xx.ravel() , yy.ravel() , zz.ravel()]).T

    data = data.astype(float)



    data = 2*(data/ 20 - 0.5)



    mask = (np.sort(np.abs(data ), axis = 1)== 1)[:,-2:].all(axis = 1)



    data = data[mask].astype(float)

    cube = data

    lim = 4
    ns = np.arange(1,lim)

    for n in ns:

        data = np.concatenate([data , data + n*2*i, data - n*2*i])

    lim = lim - 1 
    data = np.concatenate([data , cube + lim*2*i + 2*j])
    data = np.concatenate([data , cube + lim*2*i + 2*k, cube + lim*2*i + 4*k])


    # data = np.concatenate([data , cube + lim*2*i - 2*j, cube - lim*2*i - 2*j])
    

        



    # data_i = data + i

    # data_j = data + j

    # data_k = data + k

    # data = np.concatenate([data_i,data_j,data_k])






    # data = np.concatenate([data,data / 2])




    return data



def cube_faces():
    xx,yy,zz = np.indices((21,21,21))
    data = np.stack([xx.ravel() , yy.ravel() , zz.ravel()]).T

    data = data.astype(float)



    data = 2*(data/ 20 - 0.5)



    mask = (np.sort(np.abs(data ), axis = 1)== 1)[:,-3:].all(axis = 1)
    # mask2 = (np.sort(np.abs(data ), axis = 1)== 1)[:,-3:].all(axis = 1)




    data = data[mask].astype(float)

    # cube = data

    # lim = 4
    # ns = np.arange(1,lim)

    # for n in ns:

    #     data = np.concatenate([data , data + n*2*i, data - n*2*i])

    # lim = lim - 1 
    # data = np.concatenate([data , cube + lim*2*i + 2*j])
    # data = np.concatenate([data , cube + lim*2*i + 2*k, cube + lim*2*i + 4*k])
    # for n,m in ns,ns:
    #     data = np.concatenate([data , cube ])
    # data = np.concatenate([data , cube ])




        
    





    # data = np.concatenate([data , cube + lim*2*i - 2*j, cube - lim*2*i - 2*j])
    

        



    # data_i = data + i

    # data_j = data + j

    # data_k = data + k

    # data = np.concatenate([data_i,data_j,data_k])






    # data = np.concatenate([data,data / 2])




    return data


def sphere():
    xx,yy,zz = np.indices((101,101,101))
    data = np.stack([xx.ravel() , yy.ravel() , zz.ravel()]).T

    

    data = data.astype(float)
    data = 2*(data/ 100 - 0.5)


    i_axis = np.arange(30)[:,None]*i[None,:]*(0.25/30)
    j_axis = np.arange(30)[:,None]*j[None,:]*(0.5/30)
    k_axis = np.arange(30)[:,None]*k[None,:]*(0.75/30)

    

    

    mask = np.abs(lg.norm(data,axis = 1) - 1) < 0.01

    mask0 = (np.sort(np.abs(data),axis = 1)[:,:2] == 0).all(axis = 1) 

    lines = data[mask0]

    # print((np.sort(data,axis = 1)[:2] == 0).shape)

  


    data = data[mask]



    # mask2 = np.abs(data).min(axis = 1) == 0 OLD
    # mask2 = np.abs(data).min(axis = 1) == 0
    theta = (0)*np.pi*2
    i_new = np.cos(theta)*i + np.sin(theta)*j
    j_new = -np.sin(theta)*i + np.cos(theta)*j
    k_new = k

    change_basis = np.concatenate([i_new[:,None],j_new[:,None],k_new[:,None]],axis = 1)

    mask2 = np.abs(data @ change_basis).min(axis = 1) <0.01





    data = data[mask2]

    # data = np.concatenate([data,lines,i_axis,j_axis , k_axis])

    data = np.concatenate([data,-i_axis,j_axis , k_axis])



    

    # data = np.concatenate([data,np.array([[0,0,0] ])])

    
    

    # chunk = data

    # data = np.concatenate([data,chunk*1.5])
    # data = np.concatenate([data,chunk*2])

    # data2 = np.stack([xx.ravel() , yy.ravel() , zz.ravel()]).T









    # data = data[mask].astype(float)


    return data




def cylinder():
    xx,yy,zz = np.indices((101,101,101))
    data = np.stack([xx.ravel() , yy.ravel() , zz.ravel()]).T

    

    data = data.astype(float)
    data = 2*(data/ 100 - 0.5)


    i_axis = np.arange(30)[:,None]*i[None,:]*(0.25/30)
    j_axis = np.arange(30)[:,None]*j[None,:]*(0.5/30)
    k_axis = np.arange(30)[:,None]*k[None,:]*(0.75/30)

    

    mask = np.abs(lg.norm(data[:,:2],axis = 1) - 1) < 0.01
    maskx = np.abs(lg.norm(data[:,:2],axis = 1) - 1) < 0.0001


    mask_10 = (np.abs(data)[:,2] == 0) | (np.abs(data)[:,2] == 1)



    





    # mask = np.abs(lg.norm(data,axis = 1) - 1) < 0.01

    mask0 = (np.sort(np.abs(data),axis = 1)[:,:2] == 0).all(axis = 1) 

    mask20 = (np.sort(np.abs(data),axis = 1)[:,-1:] == 1).all(axis = 1) 




    # lines = data[mask0]

    # print((np.sort(data,axis = 1)[:2] == 0).shape)

  


    data = data[(maskx & mask20)|(mask & mask_10)]



    # mask2 = np.abs(data).min(axis = 1) == 0 OLD
    # mask2 = np.abs(data).min(axis = 1) == 0
    theta = (0)*np.pi*2
    i_new = np.cos(theta)*i + np.sin(theta)*j
    j_new = -np.sin(theta)*i + np.cos(theta)*j
    k_new = k

    change_basis = np.concatenate([i_new[:,None],j_new[:,None],k_new[:,None]],axis = 1)

    # mask2 = np.abs(data @ change_basis).min(axis = 1) <0.01

    mask2 = np.abs(data[:,:2] @ change_basis[:2,:]).min(axis = 1) <0.01





    data = data[mask2]

    # data = np.concatenate([data,lines,i_axis,j_axis , k_axis])

   

    # data = np.concatenate([data,data + 2*k,data - 2*k,data + 4*k,data - 4*k,data + 6*k,data - 6*k])
    # data = np.concatenate([data,data + 2*k,data - 2*k,data + 4*k,data - 4*k])
    # data = np.concatenate([data,data + 2*k,data - 2*k])
    data = np.concatenate([data])



    data = np.stack([data[:,2] , data[:,0], data[:,1]],axis = 1)
    # data = np.stack([data[:,1] , data[:,2], data[:,0]],axis = 1)




    

    # data = np.concatenate([data,np.array([[0,0,0] ])])

    
    

    # chunk = data

    # data = np.concatenate([data,chunk*1.5])
    # data = np.concatenate([data,chunk*2])

    # data2 = np.stack([xx.ravel() , yy.ravel() , zz.ravel()]).T









    # data = data[mask].astype(float)
    data = np.concatenate([data,-i_axis,j_axis , k_axis])

    return data