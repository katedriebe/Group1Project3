# Protest Activity and Perceived Violence in Photos of the Hong Kong 2019 Protests 

Group1Project3 Group Members: Katherine Driebe (kad5fwz), Ramya Tangirala (rt4an), and Lasata Tuladhar (lt9vx)

The topic that this project will be focusing on is the protest activity and perceived violence in photos of the Hong Kong 2019 protests. The dataset consists of 60 photographs from the Internet Archive, capturing various scenes and angles of the protest [1]. These images provide a visual representation of the interactions between protesters and law enforcement, scenes of civil disobedience, and public gatherings demanding democratic reforms. The link to our full data dictionary can be found here: [Data Dictionary](/DATA/Data_Dictionary.md)

Our hypothesis is as follows: **using a CNN model, images from the Hong Kong protests will depict _higher_ levels of perceived violence compared to general social media posts.** This hypothesis is informed by the expectation that protest photographs, especially those captured during significant movements like the Hong Kong protests, are likely to feature intense scenes of confrontation and civil unrest.

This project is targeted toward political figures and activists to display the severity, importance, and relevance of protests and how to amplify opinions that match their own.

## Section 1: Software and Platform
For this project, our group downloaded the `.jpg` files directly from the Internet Archive [1]. After receiving the data and downloading it into our Python script in Google Colab, we imported the following libraries: `Image` from `PIL`, `matplotlib.pyplot`, `io`, `torch`, `DataLoader` from `torch.utils.data`, `transforms` from `torchvision`, `numpy`, `pandas`, `tqdm`, `torch.nn`, `files` from `google.colab`, and `Image, Display` from `IPython.display`. All of the code was run on two different Lenovo laptops, both supporting the Windows platform, and  a Macbook Pro laptop supporting the MacOS platform.

## Section 2: Documentation Map
Our Project 3 repository is titled Group1Project3. It contains three folders (SCRIPTS, DATA, and OUTPUT), a MIT LICENSE file, and a README.md file. The SCRIPTS folder contains a master script which contains all the code needed to reproduce our data and analysis. The DATA folder contains the original dataset used for the analysis as well as the complete data dictionary. The OUTPUT folder contains all plots, including the initial exploratory plots, finals plots, and plots produced by the CNN model.

## Section 3: Reproduction of Results
Below is a step-by-step procedure of how to reproduce our group’s results:

1. Obtain the image dataset from the following link:  https://archive.org/details/HongKongProtests2019_gallery_001/EC0SY18UEAA4HnU.jpg [1]. Read in the data as the original `.jpg` type and rename the files to include the number of bytes per image.

2. Format all of the images to standardize their size and format (from `.jpg` to tensor), then normalize the data.

3. Create the CNN model from the `torch.nn` package and set it to evaluate mode.

4. Feed all of the images into the model as the inputs and set the outputs and predictions. Save the evaluation results into a `pandas` dataframe and then as a `.csv`.

5. Plot the image dataframe into categories of top, middle, and bottom entries based on `Violence`.

6. Repeat step 5 for all of the variables found in the [Data Dictionary](/DATA/Data_Dictionary.md).

7. Determine by-eye the accuracy of the CNN model based on the images.

## References (IEEE Format):
- [1] "HongKongProtests2019_gallery_001," retrieved from Archive.org, [Online]. Available: https://archive.org/details/HongKongProtests2019_gallery_001/EC0SY18UEAA4HnU.jpg. 
