from UmindNet import *
from data import *

#mydata = dataProcess(512,512)
#
#imgs_test = mydata.load_test_data()
#
#myunet = myUnet()
#
#model = myunet.get_unet()
#
#model.load_weights('unet.hdf5')
#
#imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
##
#np.save('imgs_mask_test.npy', imgs_mask_test)

from PIL import Image

#mydata = dataProcess(256,256)

dataset_dir = '/home/yang/gaolin/umind/data/test/'
def file_name(file_dir):
    L=[]
    for root,dirs,files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1]=='.tif':
                L.append(file)
    return L
my_filename = file_name(dataset_dir)
for i in my_filename:
    path = os.path.join(dataset_dir,'%s' %i)
    img = Image.open(path)
    im = img_to_array(img)
    imgdatas = np.ndarray((1,256,256,3), dtype=np.uint8)
    imgdatas[0,:,:,:]=im
    print(imgdatas.shape)
    imgdatas=imgdatas.astype('float32')
    imgdatas/=255
    myunet = myMultiUNet()
    model = myunet.MultiUnet()
    model.load_weights('multiUnet2_10.hdf5')
    imgs_mask_test = model.predict(imgdatas, verbose=1)
#    np.save('imgs_mask_test.npy', imgs_mask_test)
    imgs_mask_test = imgs_mask_test[0,:,:,:]
#    imgs_mask_test[imgs_mask_test>0.1]=255
#    imgs_mask_test[imgs_mask_test<0.1]=0
    imgs_mask_test = array_to_img(imgs_mask_test)

    imgs_mask_test.save(path[0:len(path)-4]+'umindnet2_10.jpg')