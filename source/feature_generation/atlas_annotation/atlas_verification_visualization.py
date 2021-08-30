import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

df_nodes = pd.read_csv("/home/juli/link_vessap/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_nodes_processed.csv",sep=';')
df_atlas = pd.read_csv("/home/juli/link_vessap/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_nodes_Atlas_processed.csv",sep=',')
df_filt = df_nodes.join(df_atlas)

# filter by condition

# label
# label= 'bgr'

#filt = df_filt[df_filt['Region_Acronym'] ==label]

array=np.array((df_filt))

x = array[:,1]
y = array[:,2]
z = array[:,3]

plt.title("Y/X Projection of nodes in Voreen extracted Vessel Graph")
plt.xlabel("y")
plt.ylabel("x")
plt.scatter(y,x,s=1)

plt.savefig("voreen_graph_atlas_projection")

plt.close()


## plot the registered df_atlas

                                                                                                                                      
import nibabel as nib                                                                                                                                                                                          
import numpy as np                                                                                                                                                                                             
import pandas as pd  
import matplotlib.pyplot as plt


atlas = nib.load("BALBC-no1_iso3um_stitched_atlas_registration_result.nii") 
data =  np.asanyarray(atlas.dataobj) 
data = np.abs(data)

mean = np.mean(data,axis=2)
mask = np.array(mean,dtype=bool)
plt.title("Y/X Projection of Atlas Registration Result")
plt.xlabel("y")
plt.ylabel("x")
plt.imshow(mask)
plt.savefig("atlas_projection.png")





