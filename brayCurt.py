import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.stats as stats
import matplotlib.pyplot as plt

## Global Constants
TAXA_LEVELS=['k','p','q','c','o','f','g','s']
TAXA_LEVELS_LONG=['kingdom','phylum','subphylum','class','order','family','genus','species']
TAXA_LEVELS_LONG_NOSUBPHYLUM=['kingdom','phylum','class','order','family','genus','species']

## Import Data
comboTaxa=pd.read_csv('mock10_UNITE_DITSY_101319_v2.csv') # FeatureID, Taxon, Confidence
comboPredict=pd.read_csv('mock10_deblur_repSP_table_R.csv') # FeatureID, Mock.2, Mock.1, Mock.3
expected=pd.read_table('mock10_unite_ditsy_101317_mockrobiota_expected_freq.txt') # kingdom, phylum, subphylum, class, order, family, genus, species, Mock.1, Mock.2, Mock.3
UNITE=pd.read_csv('unite_pos_sel_taxa_r.csv')
DITSY=pd.read_csv('kyria_pos_sel_taxa_r.csv')

## Reformat Taxonomy

# Initialize dictionary to store predicted taxonomy levels
levels={}
for i in TAXA_LEVELS:
    levels[i]=[]

taxa=comboTaxa['Taxon'].str.split(';')

# Seperates each row in to taxa, unkown catagorized by highest taxa level, preserves classification from just above first unkown (e.g if family is unkown, will keep order --> Unknown_f_o_Saccharomycetales)
for row in taxa:
    max_tax=0
    for key in TAXA_LEVELS:
        added=0
        for val in row:
            if val[0]==key:
                levels[key].append(val)
                added=1
        if added==0:
            if not (max_tax):
                index=TAXA_LEVELS.index(key)
                if not (index == 0):
                    max_tax=key+'_'+row[index-1]
                else:
                    max_tax=key

                levels[key].append('Unknown_'+max_tax)
            else:
                levels[key].append('Unknown_'+max_tax)

levels=pd.DataFrame(levels)
levels=levels[TAXA_LEVELS] # Order dataframe

taxa=pd.concat([levels,comboTaxa],axis=1)

taxa=taxa[['Feature ID']+TAXA_LEVELS+['Confidence']] # Order combined data base predictions

taxa.columns=[['FeatureID']+TAXA_LEVELS_LONG+['Confidence']]

predict=pd.merge(taxa,comboPredict,on='FeatureID')
predict=predict[taxa.columns.values.tolist()+['Mock.1','Mock.2','Mock.3']]
        
predict['Mock.1']= predict['Mock.1']/predict['Mock.1'].sum()
predict['Mock.2']= predict['Mock.2']/predict['Mock.2'].sum()
predict['Mock.3']= predict['Mock.3']/predict['Mock.3'].sum()

## Format Expected
Etaxa=expected[TAXA_LEVELS_LONG]

for i in range(len(Etaxa)):
    k=0
    row=Etaxa.iloc[i]
    for j,val in enumerate(row):
        if val == 'Unknown':
            repVal=val+'_'+TAXA_LEVELS[j]+'_'+row[j-1]
            Etaxa.iloc[i][j]=repVal 
expected[TAXA_LEVELS_LONG]=Etaxa
                
## Format UNITE
Upredict=pd.merge(UNITE,comboPredict,how='left',on='FeatureID')
Utaxa=Upredict[TAXA_LEVELS_LONG]
Upredict['Mock.1']= Upredict['Mock.1']/Upredict['Mock.1'].sum()
Upredict['Mock.2']= Upredict['Mock.2']/Upredict['Mock.2'].sum()
Upredict['Mock.3']= Upredict['Mock.3']/Upredict['Mock.3'].sum()

for i in range(len(Utaxa)):
    k=0
    row=Utaxa.iloc[i]
    added=0
    for j,val in enumerate(row):
        if val == 'Unknown':
            if not added:
                repVal=val+'_'+TAXA_LEVELS[j]+'_'+row[j-1]
                Utaxa.iloc[i][j]=repVal 
                added=1
            else:
                Utaxa.iloc[i][j]=repVal
UNITE[TAXA_LEVELS_LONG]=Utaxa

## Format DITSY 
Dpredict=pd.merge(DITSY,comboPredict,how='left',on='FeatureID')
Dtaxa=Dpredict[TAXA_LEVELS_LONG]
Dpredict['Mock.1']= Dpredict['Mock.1']/Dpredict['Mock.1'].sum()
Dpredict['Mock.2']= Dpredict['Mock.2']/Dpredict['Mock.2'].sum()
Dpredict['Mock.3']= Dpredict['Mock.3']/Dpredict['Mock.3'].sum()


for i in range(len(Dtaxa)):
    k=0
    row=Dtaxa.iloc[i]
    added=0
    for j,val in enumerate(row):
        if val == 'Unknown':
            if not added:
                repVal=val+'_'+TAXA_LEVELS[j]+'_'+row[j-1]
                Dtaxa.iloc[i][j]=repVal 
                added=1
            else:
                Dtaxa.iloc[i][j]=repVal
DITSY[TAXA_LEVELS_LONG]=Dtaxa
Dpredict[TAXA_LEVELS_LONG]=Dtaxa

m1=[]
m2=[]
m3=[]
Um1=[]
Um2=[]
Um3=[]
Dm1=[]
Dm2=[]
Dm3=[]

# Calculate Bray-Curtis for Hybrid,Unite,Ditsy classifiers
for level in TAXA_LEVELS_LONG_NOSUBPHYLUM: # NOT CALCULATING SUBPHYLUM
    pred=predict.groupby(level).sum()
    unite=Upredict.groupby(level).sum()
    ditsy=Dpredict.groupby(level).sum()


    exp=expected.groupby(level).sum()
    exp.columns=['Mock1E','Mock2E','Mock3E']

    merged=pd.merge(pred,exp,how='outer',right_index=True,left_index=True).fillna(0)
    Umerged=pd.merge(unite,exp,how='outer',right_index=True,left_index=True).fillna(0)
    Dmerged=pd.merge(ditsy,exp,how='outer',right_index=True,left_index=True).fillna(0)

    bc=[dist.braycurtis(merged['Mock1E'],merged['Mock.1']),dist.braycurtis(merged['Mock2E'],merged['Mock.2']),dist.braycurtis(merged['Mock3E'],merged['Mock.3'])]
    Ubc=[dist.braycurtis(Umerged['Mock1E'],Umerged['Mock.1']),dist.braycurtis(Umerged['Mock2E'],Umerged['Mock.2']),dist.braycurtis(Umerged['Mock3E'],Umerged['Mock.3'])]
    Dbc=[dist.braycurtis(Dmerged['Mock1E'],Dmerged['Mock.1']),dist.braycurtis(Dmerged['Mock2E'],Dmerged['Mock.2']),dist.braycurtis(Dmerged['Mock3E'],Dmerged['Mock.3'])]
    print(Dmerged)

    
    m1.append(bc[0])
    m2.append(bc[1])
    m3.append(bc[2])
    Um1.append(Ubc[0])
    Um2.append(Ubc[1])
    Um3.append(Ubc[2])
    Dm1.append(Dbc[0])
    Dm2.append(Dbc[1])
    Dm3.append(Dbc[2])

x=range(len(m1)) #index number for taxonomy

#unused dataframe of BC data
bcDF=pd.DataFrame({
    'x':x,
    'Mock 1':m1,
    'Mock 2':m2,
    'Mock 3':m3
})

# Lists to numpy arrays conversion
m1=np.array(m1)
m2=np.array(m2)
m3=np.array(m3)
Um1=np.array(Um1)
Um2=np.array(Um2)
Um3=np.array(Um3)
Dm1=np.array(Dm1)
Dm2=np.array(Dm2)
Dm3=np.array(Dm3)


# Plot of Bray-Curtis for Hybrid,Unite,Ditsy classifiers

fig=plt.figure

plt.plot(x,(m1+m2+m3)/3,'-r')
plt.plot(x,(Um1+Um2+Um3)/3,'-b')
plt.plot(x,(Dm1+Dm2+Dm3)/3,'-g')
plt.legend(['Hybrid','UNITE','DITSY'])

plt.scatter(x,m1,c='r')
plt.scatter(x,m2,c='r')
plt.scatter(x,m3,c='r')
plt.scatter(x,Um1,c='b')
plt.scatter(x,Um2,c='b')
plt.scatter(x,Um3,c='b')
plt.scatter(x,Dm1,c='g')
plt.scatter(x,Dm2,c='g')
plt.scatter(x,Dm3,c='g')
ax=plt.gca
plt.xticks(x,TAXA_LEVELS_LONG_NOSUBPHYLUM,rotation=20)
plt.title('Mockrobiotia Community 10 Classification Efficacy')
plt.ylabel('Bray-Curtis Distance')

plt.show()
