from dummy2 import irisSeg
import matplotlib.pyplot as plt
    #
    # you can also view using the argument in irisSeq function
    #
coord_iris, coord_pupil, output_image = irisSeg(r'C:\Users\shyam\Desktop\CNN\irisSeg-master\irisSeg\Data\shreyas.jpg', 40, 70)
print("Radius, x,y coordinates of iris: ",coord_iris) # radius and the coordinates for the center of iris 
print("Radius, x,y coordinates of pupil: ",coord_pupil) # radius and the coordinates for the center of pupil 
ratio=coord_iris[0]/coord_pupil[0]
print("Iris-Pupil ratio: ",ratio)
plt.imshow(output_image)
plt.show()
