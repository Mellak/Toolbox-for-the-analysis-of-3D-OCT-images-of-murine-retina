import os
print("start")

working_dir          = 'D:/DL_Project/3D_Project/Test_Software/'
lists_of_reina_names = os.listdir(working_dir)
for retina_name in lists_of_reina_names:
    os.system('python D:/DL_Project/Uveitis21Internship/StatisticalAnalysis/ParticleDetection/LCFCN_prediction.py ' + retina_name + ' ' + working_dir)
    os.system('python D:/DL_Project/Uveitis21Internship/StatisticalAnalysis/Extracting_Centroids/Make3DImages.py '  + retina_name + ' ' + working_dir)
    os.system('python D:/DL_Project/Uveitis21Internship/StatisticalAnalysis/RetinaDetection/Detect_Retina_Classical_Approach.py ' + retina_name + ' ' + working_dir)
    os.system('C:/Users/youne/anaconda3/python.exe D:/DL_Project/Uveitis21Internship/StatisticalAnalysis/Extracting_Centroids/3D_CC.py '+ retina_name + ' ' + working_dir)
    os.system('python D:/DL_Project/Uveitis21Internship/StatisticalAnalysis/Extracting_Centroids/detecting_centroids_4paper.py ' + retina_name + ' ' + working_dir)
    os.system('python D:/DL_Project/Uveitis21Internship/StatisticalAnalysis/Extracting_Centroids/Extract_only_1_cenctroid_4Paper.py ' + retina_name + ' ' + working_dir)
    os.system('python D:/DL_Project/Uveitis21Internship/StatisticalAnalysis/3D-KRipley/particles_retina_distance_4Paper.py ' + retina_name + ' ' + working_dir)
    #exit(1)

print("end")