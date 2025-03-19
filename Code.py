#%% Input Libraries
"""
Code for Calculating Undrained Strength Ratios based on Robertson 2020
Author : Mrunmay 
Version 1 : Jan 2024

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as mt
import statistics as st

#%% Input CPT Data and User Inputs 

loc=r'M:\Projects\US\101\00336.75\DataInfo\Calcs\CPT_2024\Undrained Strength Ratio Analyses\CLIQ results.xls '  #Address 
loc2=r'M:\Projects\US\101\00336.75\DataInfo\Calcs\CPT_2024\Undrained Strength Ratio Analyses\ '
CPTname='CPT-2412'
inputdataframe = pd.read_excel(loc, sheet_name='CPT-2412 - Liq. Results')
#CPTname='CPT-2401to07'
#inputdataframe = pd.read_excel(loc, sheet_name='CPT-01to07 ')

Phi=33    # Control Volume Friction Angle (to cap for Dilative Materials)
"""
slimes have 31 , beach have 35 . Used an average of 33 
"""
Nkt=14

#%% Dataframe Filter 
# Truncate Platform / Fill Material and NaN values 
inputdataframe=inputdataframe[inputdataframe['Ic']!=0].reset_index(drop=True)
inputdataframe=inputdataframe.dropna(subset='Ic').reset_index(drop=True)

#Initiate an output dataframe
outputdataframe=pd.DataFrame(0, columns=['Depth', 'Ic','QtnCs','qt','fs','6vo','6voef','Peak Su/P','Residual Su/P','Material Type','Behavior Type'], index=range(len(inputdataframe)))
outputdataframe['Depth']=inputdataframe['Depth (ft)']
outputdataframe['Ic']=inputdataframe['Ic']
outputdataframe['QtnCs']=inputdataframe['Qtn,cs']
outputdataframe['qt']=inputdataframe['qt (tsf)']
outputdataframe['fs']=inputdataframe['fs (tsf)']
outputdataframe['6vo']=inputdataframe.iloc[:,6]
outputdataframe['6voef']=inputdataframe.iloc[:,7]

dmin=min(outputdataframe['Depth'])-1
dmax=max(outputdataframe['Depth'])+10

#%% Calculations for Undrained Strength Ratios 

Smax=np.tan(np.radians(Phi))  # Cutoff Shear Strength Ratio for Dilative Materials 

for i in range (0,len(outputdataframe)):
    if outputdataframe['Ic'][i]<3:
        outputdataframe['Material Type'][i]="Coarse Grained"
        if outputdataframe['QtnCs'][i]<=70:
            outputdataframe['Peak Su/P'][i]=min(((6*10**-5)*outputdataframe['QtnCs'][i]**2)-0.0032*outputdataframe['QtnCs'][i]+0.2486,Smax)
            outputdataframe['Residual Su/P'][i]=(0.0007*mt.exp(0.084*outputdataframe['QtnCs'][i]))+0.3/outputdataframe['QtnCs'][i]
            outputdataframe['Behavior Type'][i]="Contractive"
        if outputdataframe['QtnCs'][i]<=82 and outputdataframe['QtnCs'][i]>70:
            outputdataframe['Peak Su/P'][i]=min((0.0007*mt.exp(0.084*outputdataframe['QtnCs'][i]))+0.3/outputdataframe['QtnCs'][i],Smax)
            outputdataframe['Residual Su/P'][i]=min((0.0007*mt.exp(0.084*outputdataframe['QtnCs'][i]))+0.3/outputdataframe['QtnCs'][i],Smax)
            outputdataframe['Behavior Type'][i]="Transitional"
        if outputdataframe['QtnCs'][i]>82:
            outputdataframe['Peak Su/P'][i]=Smax
            outputdataframe['Residual Su/P'][i]=Smax
            outputdataframe['Behavior Type'][i]="Dilative"
    if outputdataframe['Ic'][i]>=3:
        outputdataframe['Material Type'][i]="Fine Grained"
        if outputdataframe['QtnCs'][i]<=70:
            outputdataframe['Peak Su/P'][i]= min(((outputdataframe['qt'][i]-outputdataframe['6vo'][i])/outputdataframe['6voef'][i])/Nkt,Smax)
            outputdataframe['Residual Su/P'][i]=min(outputdataframe['fs'][i]/outputdataframe['6voef'][i],Smax)
            outputdataframe['Behavior Type'][i]="Contractive"
        if outputdataframe['QtnCs'][i]>70:
            outputdataframe['Peak Su/P'][i]=Smax
            outputdataframe['Residual Su/P'][i]=Smax
            outputdataframe['Behavior Type'][i]="Dilative"
            
#%% Save results to an excel file 

outputdataframe.to_excel(loc2+str(CPTname)+'_pythonResults'+'.xlsx', index=False)
            
#%% Median and Mean Calculations for Coarse Grained Peak and Residual : Including Dilative 

CoarseMaterials= outputdataframe[outputdataframe['Material Type']=='Coarse Grained'].reset_index(drop=True)
CP_Mean=round(CoarseMaterials['Peak Su/P'].mean(),2)
CP_Median=round(CoarseMaterials['Peak Su/P'].median(),2)
CP_std=round(st.stdev(CoarseMaterials['Peak Su/P']),2)

CP_dilative=len(CoarseMaterials[CoarseMaterials['Behavior Type']=='Dilative'].reset_index(drop=True))/len(CoarseMaterials)
CP_dilative=round(CP_dilative,2)
CP_contractive=len(CoarseMaterials[CoarseMaterials['Behavior Type']=='Contractive'].reset_index(drop=True))/len(CoarseMaterials)
CP_contractive=round(CP_contractive,2)
CP_transitional=len(CoarseMaterials[CoarseMaterials['Behavior Type']=='Transitional'].reset_index(drop=True))/len(CoarseMaterials)
CP_transitional=round(CP_transitional,2)

#Plot Coarse - Peak 
plt.figure(figsize=(10, 18))
plt.scatter(CoarseMaterials['Peak Su/P'], CoarseMaterials['Depth'], color='blue',edgecolor='k')
plt.axvline(x=CP_Median, color='black', linestyle='--', linewidth=4, label=f'Median={CP_Median}')
plt.axvline(x=CP_Mean+CP_std, color='red', linestyle='--', linewidth=1, label=f'Mean +/- Std. Dev.({CP_std})')
plt.axvline(x=CP_Mean-CP_std, color='red', linestyle='--', linewidth=1)
plt.axvline(x=CP_Mean, color='red', linestyle='--', linewidth=4, label=f'Mean={CP_Mean}')
plt.axvline(x=Smax, color='black', linestyle='--', linewidth=4,label='Dilative Materials Cut-off')
plt.ylim(dmin,dmax)
plt.gca().invert_yaxis()
plt.xlabel('Peak Su/P\'',fontsize=16)
plt.ylabel('Depth (ft)',fontsize=16)
plt.title('Coarse Materials (Ic<3) : Peak Undrained Strength Ratios \n Including Dilative ',size=18)
plt.legend(loc='lower right', framealpha=0.7,fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.grid(True)
plt.text(
    0.9,0.92,
    'Contractive,strain softening  = ' + str(round(CP_contractive*100,2)) + ' %\n' +
    'Dilative = ' + str(round(CP_dilative*100,2)) + ' %\n' +
    'Contractive,ductile = ' + str(round(CP_transitional*100,2)) + ' %',
    horizontalalignment='right',
    verticalalignment='bottom',  
    transform=plt.gca().transAxes,
    fontsize=16, 
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
)
plt.savefig(loc2+CPTname+'WithDil'+'Coarse_Peak.jpeg',format='jpeg',dpi=300)
plt.show()
plt.close()


CR_Mean=round(CoarseMaterials['Residual Su/P'].mean(),2)
CR_Median=round(CoarseMaterials['Residual Su/P'].median(),2)
CR_std=round(st.stdev(CoarseMaterials['Residual Su/P']),2)

CR_dilative=len(CoarseMaterials[CoarseMaterials['Behavior Type']=='Dilative'].reset_index(drop=True))/len(CoarseMaterials)
CR_dilative=round(CR_dilative,2)
CR_contractive=len(CoarseMaterials[CoarseMaterials['Behavior Type']=='Contractive'].reset_index(drop=True))/len(CoarseMaterials)
CR_contractive=round(CR_contractive,2)
CR_transitional=len(CoarseMaterials[CoarseMaterials['Behavior Type']=='Transitional'].reset_index(drop=True))/len(CoarseMaterials)
CR_transitional=round(CR_transitional,2)

#Plot Coarse - Peak 
plt.figure(figsize=(10, 18))
plt.scatter(CoarseMaterials['Residual Su/P'], CoarseMaterials['Depth'], color='blue',edgecolor='k')
plt.axvline(x=CR_Median, color='black', linestyle='--', linewidth=4, label=f'Median={CR_Median}')
plt.axvline(x=CR_Mean, color='red', linestyle='--', linewidth=4, label=f'Mean={CR_Mean}')
plt.axvline(x=CR_Mean+CR_std, color='red', linestyle='--', linewidth=1, label=f'Mean +/- Std. Dev.({CR_std})')
plt.axvline(x=CR_Mean-CR_std, color='red', linestyle='--', linewidth=1)
plt.axvline(x=Smax, color='black', linestyle='--', linewidth=2,label='Dilative Materials Cut-off')
plt.ylim(dmin,dmax)
plt.gca().invert_yaxis()
plt.xlabel('Residual Su/P\'',fontsize=16)
plt.ylabel('Depth (ft)',fontsize=16)
plt.title('Coarse Materials (Ic<3) : Residual Undrained Strength Ratios \n Including Dilative',size=18)
plt.legend(loc='lower right', framealpha=0.7,fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.grid(True)
plt.text(
    0.9,0.92,
    'Contractive,strain softening  = ' + str(round(CP_contractive*100,2)) + ' %\n' +
    'Dilative = ' + str(round(CR_dilative*100,2)) + ' %\n' +
    'Contractive,ductile = ' + str(round(CR_transitional*100,2)) + ' %',
    horizontalalignment='right',
    verticalalignment='bottom',  
    transform=plt.gca().transAxes,
    fontsize=16, 
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
)
plt.savefig(loc2+CPTname+'WithDil'+'Coarse_Residual.jpeg',format='jpeg',dpi=300)
plt.show()
plt.close()




#%% Median and Mean Calculations for Fine Grained Peak and Residual : Including Dilative

FineMaterials= outputdataframe[outputdataframe['Material Type']=='Fine Grained'].reset_index(drop=True)
FP_Mean=round(FineMaterials['Peak Su/P'].mean(),2)
FP_Median=round(FineMaterials['Peak Su/P'].median(),2)
FP_std=round(st.stdev(FineMaterials['Peak Su/P']),2)

FP_dilative=len(FineMaterials[FineMaterials['Behavior Type']=='Dilative'].reset_index(drop=True))/len(FineMaterials)
FP_dilative=round(FP_dilative,2)
FP_contractive=len(FineMaterials[FineMaterials['Behavior Type']=='Contractive'].reset_index(drop=True))/len(FineMaterials)
FP_contractive=round(FP_contractive,2)


#Plot Fine
plt.figure(figsize=(10, 18))
plt.scatter(FineMaterials['Peak Su/P'], FineMaterials['Depth'], color='red',edgecolor='k')
plt.axvline(x=FP_Median, color='black', linestyle='--', linewidth=4, label=f'Median={FP_Median}')
plt.axvline(x=FP_Mean, color='red', linestyle='--', linewidth=4, label=f'Mean={FP_Mean}')
plt.axvline(x=FP_Mean+FP_std, color='red', linestyle='--', linewidth=1, label=f'Mean +/- Std. Dev.({FP_std})')
plt.axvline(x=FP_Mean-FP_std, color='red', linestyle='--', linewidth=1)
plt.axvline(x=Smax, color='black', linestyle='--', linewidth=2,label='Dilative Materials Cut-off')
plt.ylim(dmin,dmax)
plt.gca().invert_yaxis()
plt.xlabel('Peak Su/P\'',fontsize=16)
plt.ylabel('Depth (ft)',fontsize=16)
plt.title('Fine Materials (Ic>=3) : Peak Undrained Strength Ratios \n Including Dilative',size=18)
plt.legend(loc='lower right', framealpha=0.7,fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.grid(True)
plt.text(
    0.9,0.92,
    'Contractive,strain softening  = ' + str(FP_contractive*100) + ' %\n' +
    'Dilative = ' + str(FP_dilative*100) + ' %\n' ,
    #'Contractive,ductile = ' + str(CP_transitional*100) + ' %',
    horizontalalignment='right',
    verticalalignment='bottom',  
    transform=plt.gca().transAxes,
    fontsize=16, 
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
)
plt.savefig(loc2+CPTname+'WithDil'+'Fines_Peak.jpeg',format='jpeg',dpi=300)
plt.show()
plt.close()


FR_Mean=round(FineMaterials['Residual Su/P'].mean(),2)
FR_Median=round(FineMaterials['Residual Su/P'].median(),2)
FR_std=round(st.stdev(FineMaterials['Residual Su/P']),2)


FR_dilative=len(FineMaterials[FineMaterials['Behavior Type']=='Dilative'].reset_index(drop=True))/len(FineMaterials)
FR_dilative=round(FR_dilative,2)
FR_contractive=len(FineMaterials[FineMaterials['Behavior Type']=='Contractive'].reset_index(drop=True))/len(FineMaterials)
FR_contractive=round(FR_contractive,2)

#Plot Fine
plt.figure(figsize=(10, 18))
plt.scatter(FineMaterials['Residual Su/P'], FineMaterials['Depth'], color='red',edgecolor='k')
plt.axvline(x=FR_Median, color='black', linestyle='--', linewidth=4, label=f'Median={FR_Median}')
plt.axvline(x=FR_Mean, color='red', linestyle='--', linewidth=4, label=f'Mean={FR_Mean}')
plt.axvline(x=FR_Mean+FR_std, color='red', linestyle='--', linewidth=1, label=f'Mean +/- Std. Dev.({FR_std})')
plt.axvline(x=FR_Mean-FR_std, color='red', linestyle='--', linewidth=1)
plt.axvline(x=Smax, color='black', linestyle='--', linewidth=2,label='Dilative Materials Cut-off')
plt.ylim(dmin,dmax)
plt.gca().invert_yaxis()
plt.xlabel('Residual Su/P\'',fontsize=16)
plt.ylabel('Depth (ft)',fontsize=16)
plt.title('Fine Materials (Ic>=3) : Residual Undrained Strength Ratios \n Including Dilative',size=18)
plt.legend(loc='lower right', framealpha=0.7,fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.grid(True)
plt.text(
    0.9,0.92,
    'Contractive,strain softening  = ' + str(FR_contractive*100) + ' %\n' +
    'Dilative = ' + str(FR_dilative*100) + ' %\n' ,
    #'Contractive,ductile = ' + str(CP_transitional*100) + ' %',
    horizontalalignment='right',
    verticalalignment='bottom',  
    transform=plt.gca().transAxes,
    fontsize=16, 
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
)
plt.savefig(loc2+CPTname+'WithDil'+'Fines_Residual.jpeg',format='jpeg',dpi=300)
plt.show()
plt.close()


#%% Median and Mean Calculations for Combined Fine and Coarse : Including Dilative

CP_Mean=round(outputdataframe['Peak Su/P'].mean(),2)
CP_Median=round(outputdataframe['Peak Su/P'].median(),2)
CP_std=round(st.stdev(outputdataframe['Peak Su/P']),2)

CP_dilative=len(outputdataframe[outputdataframe['Behavior Type']=='Dilative'].reset_index(drop=True))/len(outputdataframe)
CP_dilative=round(CP_dilative,2)
CP_contractive=len(outputdataframe[outputdataframe['Behavior Type']=='Contractive'].reset_index(drop=True))/len(outputdataframe)
CP_contractive=round(CP_contractive,2)
CP_transitional=len(outputdataframe[outputdataframe['Behavior Type']=='Transitional'].reset_index(drop=True))/len(outputdataframe)
CP_transitional=round(CP_transitional,2)

coarseMaterials= outputdataframe[outputdataframe['Material Type']=='Coarse Grained'].reset_index(drop=True)
FineMaterials= outputdataframe[outputdataframe['Material Type']=='Fine Grained'].reset_index(drop=True)

#Plot Coarse - Peak 
plt.figure(figsize=(10, 18))
plt.scatter(CoarseMaterials['Peak Su/P'], CoarseMaterials['Depth'], color='blue',edgecolor='k')
plt.scatter(FineMaterials['Peak Su/P'], FineMaterials['Depth'], color='red',edgecolor='k')
plt.axvline(x=CP_Median, color='black', linestyle='--', linewidth=4, label=f'Median={CP_Median}')
plt.axvline(x=CP_Mean, color='red', linestyle='--', linewidth=4, label=f'Mean={CP_Mean}')
plt.axvline(x=CP_Mean+CP_std, color='red', linestyle='--', linewidth=1, label=f'Mean +/- Std. Dev.({CP_std})')
plt.axvline(x=CP_Mean-CP_std, color='red', linestyle='--', linewidth=1)
plt.axvline(x=Smax, color='black', linestyle='--', linewidth=2,label='Dilative Materials Cut-off')
plt.ylim(dmin,dmax)
plt.gca().invert_yaxis()
plt.xlabel('Peak Su/P\'',fontsize=16)
plt.ylabel('Depth (ft)',fontsize=16)
plt.title('Coarse and Fine Combined Materials : Peak Undrained Strength Ratios \n Including Dilative ',size=18)
plt.legend(loc='lower right', framealpha=0.7,fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.grid(True)
plt.text(
    0.9,0.92,
    'Contractive,strain softening  = ' + str(round(CP_contractive*100,2)) + ' %\n' +
    'Dilative = ' + str(round(CP_dilative*100,2)) + ' %\n' +
    'Contractive,ductile = ' + str(round(CP_transitional*100,2)) + ' %',
    horizontalalignment='right',
    verticalalignment='bottom',  
    transform=plt.gca().transAxes,
    fontsize=16, 
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
)
plt.savefig(loc2+CPTname+'WithDil'+'Combined_Peak.jpeg',format='jpeg',dpi=300)
plt.show()
plt.close()


CR_Mean=round(outputdataframe['Residual Su/P'].mean(),2)
CR_Median=round(outputdataframe['Residual Su/P'].median(),2)
CR_std=round(st.stdev(outputdataframe['Residual Su/P']),2)


CR_dilative=len(outputdataframe[outputdataframe['Behavior Type']=='Dilative'].reset_index(drop=True))/len(outputdataframe)
CR_dilative=round(CR_dilative,2)
CR_contractive=len(outputdataframe[outputdataframe['Behavior Type']=='Contractive'].reset_index(drop=True))/len(outputdataframe)
CR_contractive=round(CR_contractive,2)
CR_transitional=len(outputdataframe[outputdataframe['Behavior Type']=='Transitional'].reset_index(drop=True))/len(outputdataframe)
CR_transitional=round(CR_transitional,2)

#Plot Coarse - Peak 
plt.figure(figsize=(10, 18))
plt.scatter(CoarseMaterials['Residual Su/P'], CoarseMaterials['Depth'], color='blue',edgecolor='k')
plt.scatter(FineMaterials['Residual Su/P'], FineMaterials['Depth'], color='red',edgecolor='k')
plt.axvline(x=CR_Median, color='black', linestyle='--', linewidth=4, label=f'Median={CR_Median}')
plt.axvline(x=CR_Mean, color='red', linestyle='--', linewidth=4, label=f'Mean={CR_Mean}')
plt.axvline(x=CR_Mean+CR_std, color='red', linestyle='--', linewidth=1, label=f'Mean +/- Std. Dev.({CR_std})')
plt.axvline(x=CR_Mean-CR_std, color='red', linestyle='--', linewidth=1)
plt.axvline(x=Smax, color='black', linestyle='--', linewidth=2,label='Dilative Materials Cut-off')
plt.ylim(dmin,dmax)
plt.gca().invert_yaxis()
plt.xlabel('Residual Su/P\'',fontsize=16)
plt.ylabel('Depth (ft)',fontsize=16)
plt.title('Coarse and Fine Combined Materials : Residual Undrained Strength Ratios \n Including Dilative',size=18)
plt.legend(loc='lower right', framealpha=0.7,fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.grid(True)
plt.text(
    0.9,0.92,
    'Contractive,strain softening  = ' + str(round(CP_contractive*100,2)) + ' %\n' +
    'Dilative = ' + str(round(CR_dilative*100,2)) + ' %\n' +
    'Contractive,ductile = ' + str(round(CR_transitional*100,2)) + ' %',
    horizontalalignment='right',
    verticalalignment='bottom',  
    transform=plt.gca().transAxes,
    fontsize=16, 
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
)
plt.savefig(loc2+CPTname+'WithDil'+'Combined_Residual.jpeg',format='jpeg',dpi=300)
plt.show()
plt.close()







#%% Median and Mean Calculations for Coarse Grained Peak and Residual : Excluding Dilative 

CoarseMaterials= outputdataframe[outputdataframe['Material Type']=='Coarse Grained'].reset_index(drop=True)
CoarseMaterials= CoarseMaterials[CoarseMaterials['Peak Su/P']<Smax].reset_index(drop=True)
CP_Mean=round(CoarseMaterials['Peak Su/P'].mean(),2)
CP_Median=round(CoarseMaterials['Peak Su/P'].median(),2)
CP_std=round(st.stdev(CoarseMaterials['Peak Su/P']),2)

CP_dilative=len(CoarseMaterials[CoarseMaterials['Behavior Type']=='Dilative'].reset_index(drop=True))/len(CoarseMaterials)
CP_dilative=round(CP_dilative,2)
CP_contractive=len(CoarseMaterials[CoarseMaterials['Behavior Type']=='Contractive'].reset_index(drop=True))/len(CoarseMaterials)
CP_contractive=round(CP_contractive,2)
CP_transitional=len(CoarseMaterials[CoarseMaterials['Behavior Type']=='Transitional'].reset_index(drop=True))/len(CoarseMaterials)
CP_transitional=round(CP_transitional,2)

#Plot Coarse - Peak 
plt.figure(figsize=(10, 18))
plt.scatter(CoarseMaterials['Peak Su/P'], CoarseMaterials['Depth'], color='blue',edgecolor='k')
plt.axvline(x=CP_Median, color='black', linestyle='--', linewidth=4, label=f'Median={CP_Median}')
plt.axvline(x=CP_Mean, color='red', linestyle='--', linewidth=4, label=f'Mean={CP_Mean}')
plt.axvline(x=CP_Mean+CP_std, color='red', linestyle='--', linewidth=1, label=f'Mean +/- Std. Dev.({CP_std})')
plt.axvline(x=CP_Mean-CP_std, color='red', linestyle='--', linewidth=1)
plt.axvline(x=Smax, color='black', linestyle='--', linewidth=2,label='Dilative Materials Cut-off')
plt.ylim(dmin,dmax)
plt.gca().invert_yaxis()
plt.xlabel('Peak Su/P\'',fontsize=16)
plt.ylabel('Depth (ft)',fontsize=16)
plt.title('Coarse Materials (Ic<3) : Peak Undrained Strength Ratios \n Excluding Dilative ',size=18)
plt.legend(loc='lower right', framealpha=0.7,fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.grid(True)
plt.text(
    0.9,0.92,
    'Contractive,strain softening  = ' + str(round(CP_contractive*100,2)) + ' %\n' +
    'Dilative = ' + str(CP_dilative*100) + ' %\n' +
    'Contractive,ductile = ' + str(CP_transitional*100) + ' %',
    horizontalalignment='right',
    verticalalignment='bottom',  
    transform=plt.gca().transAxes,
    fontsize=16, 
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
)
plt.savefig(loc2+CPTname+'ExDil'+'Coarse_Peak.jpeg',format='jpeg',dpi=300)
plt.show()
plt.close()


CR_Mean=round(CoarseMaterials['Residual Su/P'].mean(),2)
CR_Median=round(CoarseMaterials['Residual Su/P'].median(),2)
CR_std=round(st.stdev(CoarseMaterials['Residual Su/P']),2)

CR_dilative=len(CoarseMaterials[CoarseMaterials['Behavior Type']=='Dilative'].reset_index(drop=True))/len(CoarseMaterials)
CR_dilative=round(CR_dilative,2)
CR_contractive=len(CoarseMaterials[CoarseMaterials['Behavior Type']=='Contractive'].reset_index(drop=True))/len(CoarseMaterials)
CR_contractive=round(CR_contractive,2)
CR_transitional=len(CoarseMaterials[CoarseMaterials['Behavior Type']=='Transitional'].reset_index(drop=True))/len(CoarseMaterials)
CR_transitional=round(CR_transitional,2)

#Plot Coarse - Peak 
plt.figure(figsize=(10, 18))
plt.scatter(CoarseMaterials['Residual Su/P'], CoarseMaterials['Depth'], color='blue',edgecolor='k')
plt.axvline(x=CR_Median, color='black', linestyle='--', linewidth=4, label=f'Median={CR_Median}')
plt.axvline(x=CR_Mean, color='red', linestyle='--', linewidth=4, label=f'Mean={CR_Mean}')
plt.axvline(x=CR_Mean+CR_std, color='red', linestyle='--', linewidth=1, label=f'Mean +/- Std. Dev.({CR_std})')
plt.axvline(x=CR_Mean-CR_std, color='red', linestyle='--', linewidth=1)
plt.axvline(x=Smax, color='black', linestyle='--', linewidth=2,label='Dilative Materials Cut-off')
plt.ylim(dmin,dmax)
plt.gca().invert_yaxis()
plt.xlabel('Residual Su/P\'',fontsize=16)
plt.ylabel('Depth (ft)',fontsize=16)
plt.title('Coarse Materials (Ic<3) : Residual Undrained Strength Ratios \n Excluding Dilative',size=18)
plt.legend(loc='lower right', framealpha=0.7,fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.grid(True)
plt.text(
    0.9,0.92,
    'Contractive,strain softening  = ' + str(round(CP_contractive*100,2)) + ' %\n' +
    'Dilative = ' + str(CR_dilative*100) + ' %\n' +
    'Contractive,ductile = ' + str(CR_transitional*100) + ' %',
    horizontalalignment='right',
    verticalalignment='bottom',  
    transform=plt.gca().transAxes,
    fontsize=16, 
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
)
plt.savefig(loc2+CPTname+'ExDil'+'Coarse_Residual.jpeg',format='jpeg',dpi=300)
plt.show()
plt.close()

#%% Median and Mean Calculations for Fine Grained Peak and Residual : Excluding Dilative

FineMaterials= outputdataframe[outputdataframe['Material Type']=='Fine Grained'].reset_index(drop=True)
FineMaterials1= FineMaterials[FineMaterials['Peak Su/P']<Smax].reset_index(drop=True)

FP_Mean=round(FineMaterials1['Peak Su/P'].mean(),2)
FP_Median=round(FineMaterials1['Peak Su/P'].median(),2)
FP_std=round(st.stdev(FineMaterials1['Peak Su/P']),2)

FP_dilative=len(FineMaterials1[FineMaterials1['Behavior Type']=='Dilative'].reset_index(drop=True))/len(FineMaterials1)
FP_dilative=round(FP_dilative,2)
FP_contractive=len(FineMaterials1[FineMaterials1['Behavior Type']=='Contractive'].reset_index(drop=True))/len(FineMaterials1)
FP_contractive=round(FP_contractive,2)

#Plot Fine
plt.figure(figsize=(10, 18))
plt.scatter(FineMaterials1['Peak Su/P'], FineMaterials1['Depth'], color='red',edgecolor='k')
plt.axvline(x=FP_Median, color='black', linestyle='--', linewidth=4, label=f'Median={FP_Median}')
plt.axvline(x=FP_Mean, color='red', linestyle='--', linewidth=4, label=f'Mean={FP_Mean}')
plt.axvline(x=FP_Mean+FP_std, color='red', linestyle='--', linewidth=1, label=f'Mean +/- Std. Dev.({FP_std})')
plt.axvline(x=FP_Mean-FP_std, color='red', linestyle='--', linewidth=1)
plt.axvline(x=Smax, color='black', linestyle='--', linewidth=2,label='Dilative Materials Cut-off')
plt.ylim(dmin,dmax)
plt.gca().invert_yaxis()
plt.xlabel('Peak Su/P\'',fontsize=16)
plt.ylabel('Depth (ft)',fontsize=16)
plt.title('Fine Materials (Ic>=3) : Peak Undrained Strength Ratios \n Excluding Dilative',size=18)
plt.legend(loc='lower right', framealpha=0.7,fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.grid(True)
plt.text(
    0.9,0.92,
    'Contractive,strain softening  = ' + str(FP_contractive*100) + ' %\n' +
    'Dilative = ' + str(FP_dilative*100) + ' %\n' ,
    #'Contractive,ductile = ' + str(CP_transitional*100) + ' %',
    horizontalalignment='right',
    verticalalignment='bottom',  
    transform=plt.gca().transAxes,
    fontsize=16, 
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
)
plt.savefig(loc2+CPTname+'ExDil'+'Fines_Peak.jpeg',format='jpeg',dpi=300)
plt.show()
plt.close()


FineMaterials2= FineMaterials[FineMaterials['Residual Su/P']<Smax].reset_index(drop=True)

FR_Mean=round(FineMaterials2['Residual Su/P'].mean(),2)
FR_Median=round(FineMaterials2['Residual Su/P'].median(),2)
FR_std=round(st.stdev(FineMaterials2['Residual Su/P']),2)

FR_dilative=len(FineMaterials2[FineMaterials2['Behavior Type']=='Dilative'].reset_index(drop=True))/len(FineMaterials2)
FR_dilative=round(FR_dilative,2)
FR_contractive=len(FineMaterials2[FineMaterials2['Behavior Type']=='Contractive'].reset_index(drop=True))/len(FineMaterials2)
FR_contractive=round(FR_contractive,2)

#Plot Fine
plt.figure(figsize=(10, 18))
plt.scatter(FineMaterials2['Residual Su/P'], FineMaterials2['Depth'], color='red',edgecolor='k')
plt.axvline(x=FR_Median, color='black', linestyle='--', linewidth=4, label=f'Median={FR_Median}')
plt.axvline(x=FR_Mean, color='red', linestyle='--', linewidth=4, label=f'Mean={FR_Mean}')
plt.axvline(x=FR_Mean+FR_std, color='red', linestyle='--', linewidth=1, label=f'Mean +/- Std. Dev.({FR_std})')
plt.axvline(x=FR_Mean-FR_std, color='red', linestyle='--', linewidth=1)
plt.axvline(x=Smax, color='black', linestyle='--', linewidth=2,label='Dilative Materials Cut-off')
plt.ylim(dmin,dmax)
plt.gca().invert_yaxis()
plt.xlabel('Residual Su/P\'',fontsize=16)
plt.ylabel('Depth (ft)',fontsize=16)
plt.title('Fine Materials (Ic>=3) : Residual Undrained Strength Ratios \n Excluding Dilative',size=18)
plt.legend(loc='lower right', framealpha=0.7,fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.grid(True)
plt.text(
    0.9,0.92,
    'Contractive,strain softening  = ' + str(FR_contractive*100) + ' %\n' +
    'Dilative = ' + str(FR_dilative*100) + ' %\n' ,
    #'Contractive,ductile = ' + str(CP_transitional*100) + ' %',
    horizontalalignment='right',
    verticalalignment='bottom',  
    transform=plt.gca().transAxes,
    fontsize=16, 
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
)
plt.savefig(loc2+CPTname+'ExDil'+'Fines_Residual.jpeg',format='jpeg',dpi=300)
plt.show()
plt.close()









#%% Median and Mean Calculations for Combined Fine and Coarse : Excluding Dilative 

outputdataframe=outputdataframe[outputdataframe['Peak Su/P']<Smax]

CP_Mean=round(outputdataframe['Peak Su/P'].mean(),2)
CP_Median=round(outputdataframe['Peak Su/P'].median(),2)
CP_std=round(st.stdev(outputdataframe['Peak Su/P']),2)

CP_dilative=len(outputdataframe[outputdataframe['Behavior Type']=='Dilative'].reset_index(drop=True))/len(outputdataframe)
CP_dilative=round(CP_dilative,2)
CP_contractive=len(outputdataframe[outputdataframe['Behavior Type']=='Contractive'].reset_index(drop=True))/len(outputdataframe)
CP_contractive=round(CP_contractive,2)
CP_transitional=len(outputdataframe[outputdataframe['Behavior Type']=='Transitional'].reset_index(drop=True))/len(outputdataframe)
CP_transitional=round(CP_transitional,2)

coarseMaterials= outputdataframe[outputdataframe['Material Type']=='Coarse Grained'].reset_index(drop=True)
FineMaterials= outputdataframe[outputdataframe['Material Type']=='Fine Grained'].reset_index(drop=True)

#Plot Coarse - Peak 
plt.figure(figsize=(10, 18))
plt.scatter(CoarseMaterials['Peak Su/P'], CoarseMaterials['Depth'], color='blue',edgecolor='k')
plt.scatter(FineMaterials['Peak Su/P'], FineMaterials['Depth'], color='red',edgecolor='k')
plt.axvline(x=CP_Median, color='black', linestyle='--', linewidth=4, label=f'Median={CP_Median}')
plt.axvline(x=CP_Mean, color='red', linestyle='--', linewidth=4, label=f'Mean={CP_Mean}')
plt.axvline(x=CP_Mean+CP_std, color='red', linestyle='--', linewidth=1, label=f'Mean +/- Std. Dev.({CP_std})')
plt.axvline(x=CP_Mean-CP_std, color='red', linestyle='--', linewidth=1)
plt.axvline(x=Smax, color='black', linestyle='--', linewidth=2,label='Dilative Materials Cut-off')
plt.ylim(dmin,dmax)
plt.gca().invert_yaxis()
plt.xlabel('Peak Su/P\'',fontsize=16)
plt.ylabel('Depth (ft)',fontsize=16)
plt.title('Coarse and Fine Combined Materials : Peak Undrained Strength Ratios \n Excluding Dilative ',size=18)
plt.legend(loc='lower right', framealpha=0.7,fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.grid(True)
plt.text(
    0.9,0.92,
    'Contractive,strain softening  = ' + str(round(CP_contractive*100,2)) + ' %\n' +
    'Dilative = ' + str(round(CP_dilative*100,2)) + ' %\n' +
    'Contractive,ductile = ' + str(round(CP_transitional*100,2)) + ' %',
    horizontalalignment='right',
    verticalalignment='bottom',  
    transform=plt.gca().transAxes,
    fontsize=16, 
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
)
plt.savefig(loc2+CPTname+'ExDil'+'Combined_Peak.jpeg',format='jpeg',dpi=300)
plt.show()
plt.close()


CR_Mean=round(outputdataframe['Residual Su/P'].mean(),2)
CR_Median=round(outputdataframe['Residual Su/P'].median(),2)
CR_std=round(st.stdev(outputdataframe['Residual Su/P']),2)

CR_dilative=len(outputdataframe[outputdataframe['Behavior Type']=='Dilative'].reset_index(drop=True))/len(outputdataframe)
CR_dilative=round(CR_dilative,2)
CR_contractive=len(outputdataframe[outputdataframe['Behavior Type']=='Contractive'].reset_index(drop=True))/len(outputdataframe)
CR_contractive=round(CR_contractive,2)
CR_transitional=len(outputdataframe[outputdataframe['Behavior Type']=='Transitional'].reset_index(drop=True))/len(outputdataframe)
CR_transitional=round(CR_transitional,2)

#Plot Coarse - Peak 
plt.figure(figsize=(10, 18))
plt.scatter(CoarseMaterials['Residual Su/P'], CoarseMaterials['Depth'], color='blue',edgecolor='k')
plt.scatter(FineMaterials['Residual Su/P'], FineMaterials['Depth'], color='red',edgecolor='k')
plt.axvline(x=CR_Median, color='black', linestyle='--', linewidth=4, label=f'Median={CR_Median}')
plt.axvline(x=CR_Mean, color='red', linestyle='--', linewidth=4, label=f'Mean={CR_Mean}')
plt.axvline(x=CR_Mean+CR_std, color='red', linestyle='--', linewidth=1, label=f'Mean +/- Std. Dev.({CR_std})')
plt.axvline(x=CR_Mean-CR_std, color='red', linestyle='--', linewidth=1)
plt.axvline(x=Smax, color='black', linestyle='--', linewidth=2,label='Dilative Materials Cut-off')
plt.ylim(dmin,dmax)
plt.gca().invert_yaxis()
plt.xlabel('Residual Su/P\'',fontsize=16)
plt.ylabel('Depth (ft)',fontsize=16)
plt.title('Coarse and Fine Combined Materials : Residual Undrained Strength Ratios \n Excluding Dilative',size=18)
plt.legend(loc='lower right', framealpha=0.7,fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.grid(True)
plt.text(
    0.9,0.92,
    'Contractive,strain softening  = ' + str(round(CP_contractive*100,2)) + ' %\n' +
    'Dilative = ' + str(round(CR_dilative*100,2)) + ' %\n' +
    'Contractive,ductile = ' + str(round(CR_transitional*100,2)) + ' %',
    horizontalalignment='right',
    verticalalignment='bottom',  
    transform=plt.gca().transAxes,
    fontsize=16, 
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
)
plt.savefig(loc2+CPTname+'ExDil'+'Combined_Residual.jpeg',format='jpeg',dpi=300)
plt.show()
plt.close()









































