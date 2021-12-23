
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

FOREGROUND_THRESHOLD = 0.25

def sanity_check(patch_size,img_predict_patch):
    plt.figure(figsize=(9, 9))
    if(patch_size == 128):
      square = 5
    if(patch_size == 256):
      square = 3
    ix = 1
    i=2
    for k in range(square):
	    for j in range(square):
		    # specify subplot and turn of axis
		    ax = plt.subplot(square, square, ix)
		    ax.set_xticks([])
		    ax.set_yticks([])
		    # plot 
		    plt.imshow(img_predict_patch[i, j, k, :, :], cmap='Greys_r')
		    ix += 1
    # show the figure
    plt.show()


def prediction(img_patches, model):
    img_predict_patch = []
    for i in range(img_patches.shape[0]):
      for j in range(img_patches.shape[1]):
        for k in range(img_patches.shape[2]):
            print("Now predicting on patch", i,j,k)

            patch = img_patches[i,j,k,:,:,:]
            patch_input=np.expand_dims(patch, 0)
            patch_prediction = model.predict(patch_input)
 
            img_predict_patch.append(patch_prediction)

    return img_predict_patch

def multi_prediciton(img_patches1, img_patches2, img_patches3, model1,model2,model3):
    img_predict_patch = []
    for i in range(img_patches1.shape[0]):
      for j in range(img_patches1.shape[1]):
        for k in range(img_patches1.shape[2]):
            print("Now predicting on patch", i,j,k)

            patch1 = img_patches1[i,j,k,:,:,:]
            patch2 = img_patches2[i,j,k,:,:,:] 
            patch3 = img_patches3[i,j,k,:,:,:] 
            
            patch_input1=np.expand_dims(patch1, 0)
            patch_input2=np.expand_dims(patch2, 0)
            patch_input3=np.expand_dims(patch3, 0)

        
            patch_prediction1 = model1.predict(patch_input1)
            patch_prediction2 = model2.predict(patch_input2)
            patch_prediction3 = model3.predict(patch_input3)

            #calculate mean of 3 predictions
            patch_prediction = np.asarray((patch_prediction1+patch_prediction2+patch_prediction3)/3)
            img_predict_patch.append(patch_prediction)
    return img_predict_patch


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# assign a label to a patch
def patch_to_label(patch):
    """Maps a BW white patch image to a label using thresholding """
    df = np.mean(patch)
    if df > FOREGROUND_THRESHOLD:
        return 1
    else:
        return 0
    

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data 


def img_crop(im, patch_size, padding):
    """ From a test image, creates patches of height and width : window_size every patch_size pixels"""
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    #The test image needs to be padded to create windows with a center on the edge of the image
    im = np.lib.pad(im, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    for i in range(padding, imgheight + padding, patch_size):
        for j in range(padding, imgwidth + padding, patch_size):
            #creation of the window surrounding the patch
            im_patch = im[j - padding:j + patch_size + padding, i - padding:i + patch_size + padding, :]
            list_patches.append(im_patch)
    return list_patches



def gen_patches(imgs, window_size, patch_size):
    """Generate patches from the set of test images"""
    padding_size = (window_size - patch_size) // 2
    patches = np.asarray(
        [img_crop(imgs[i], patch_size, padding_size) for i in  range(imgs.shape[0])])
    return patches.reshape(-1, patches.shape[2], patches.shape[3], patches.shape[4])


def mask_to_submission_strings_CNN(model,image_filename,window_size,patch_size,num_images=50):
    """Create patches  and predict the class of each patch"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    image = load_image(image_filename)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #create windows of size window_size surrounding patches of size patch_size 
    patches = gen_patches(image,window_size, patch_size)
    labels = model.predict(patches)
    labels = (labels[:,0]<labels[:,1])*1
    count = 0
    print("Processing image => " + image_filename)
    for j in range(0, image.shape[2], patch_size):
        for i in range(0, image.shape[1], patch_size):
            label = int(labels[count])
            count += 1
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))

            
def mask_to_submission_strings_unet(image_filename):
    """Reads a predicted image (BW) and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i: i + patch_size, j: j + patch_size]
            label = patch_to_label(patch)
            yield "{:03d}_{}_{},{}".format(img_number, j, i, label)

def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings_unet(fn))
            
def masks_to_submission_cnn(model, submission_path, *image_filenames,window_size,patch_size):
    """ Generate a .csv containing the classification of the test set. """
    with open(submission_path, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
             f.writelines('{}\n'.format(s) for s in mask_to_submission_strings_CNN(model, fn,window_size,patch_size))

            