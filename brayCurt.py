import pandas as pd
import scipy.stats as stats
import scipy.spatial.distance as dist
import numpy as np
import matplotlib.pyplot as plt
## Global Variables
TAXA_LEVELS=['k','p','q','c','o','f','g','s']
TAXA_LEVELS_LONG=['kingdom','phylum','subphylum','class','order','family','genus','species']

## Import Data
comboTaxa=pd.read_csv('mock10_UNITE_DITSY_101319_v2.csv') # FeatureID, Taxon, Confidence
comboPredict=pd.read_csv('mock10_deblur_repSP_table_R.csv') # FeatureID, Mock.2, Mock.1, Mock.3
expected=pd.read_table('mock10_unite_ditsy_101317_mockrobiota_expected_freq.txt') # kingdom, phylum, subphylum, class, order, family, genus, species, Mock.1, Mock.2, Mock.3

## Reformat Taxonomy

# Initialize dictionary to store predicted taxonomy levels
levels={}
for i in TAXA_LEVELS:
    levels[i]=[]

taxa=comboTaxa['Taxon'].str.split(';')

# Seperates each row in to taxa, unkown catagorized by highest taxa level
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

predict=pd.merge(taxa,comboPredict,how='left',on='FeatureID')
predict=predict[taxa.columns.values.tolist()+['Mock.1','Mock.2','Mock.3']]
        
predict['Mock.1']= predict['Mock.1']/predict['Mock.1'].sum()
predict['Mock.2']= predict['Mock.2']/predict['Mock.2'].sum()
predict['Mock.3']= predict['Mock.3']/predict['Mock.3'].sum()

## Format Expected
Etaxa=expected[TAXA_LEVELS_LONG]
Elevels={}
for i in TAXA_LEVELS:
    Elevels[i]=[]

for i in range(len(Etaxa)):
    k=0
    row=Etaxa.iloc[i]
    for j,val in enumerate(row):
        if val == 'Unknown':
            repVal=val+'_'+TAXA_LEVELS[j]+'_'+row[j-1]
            Etaxa.iloc[i][j]=repVal 
expected[TAXA_LEVELS_LONG]=Etaxa
                


m1=[]
m2=[]
m3=[]

for level in TAXA_LEVELS_LONG:
    pred=predict.groupby(level).sum()

    exp=expected.groupby(level).sum()
    exp.columns=['Mock1E','Mock2E','Mock3E']

    merged=pd.merge(pred,exp,how='outer',right_index=True,left_index=True).fillna(0)
    bc=[dist.braycurtis(merged['Mock1E'],merged['Mock.1']),dist.braycurtis(merged['Mock2E'],merged['Mock.2']),dist.braycurtis(merged['Mock3E'],merged['Mock.3'])]


    m1.append(bc[0])
    m2.append(bc[1])
    m3.append(bc[2])

x=range(len(m1))

bcDF=pd.DataFrame({
    'x':x,
    'Mock 1':m1,
    'Mock 2':m2,
    'Mock 3':m3
})



fig=plt.figure
plt.scatter(x,m1,c='r')
plt.scatter(x,m2,c='r')
plt.scatter(x,m3,c='r')

plt.title('Bray-Curtis Distance Replaced SubSpecies')
plt.show()

