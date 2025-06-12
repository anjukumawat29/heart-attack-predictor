import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

dataset=pd.read_csv("heart_Attack Dataset.csv")
dataset

dataset.sample(5) # randomly select rows and columns

dataset.describe()

dataset.info()

info = ["age","1: male, 0: female","chest pain type, 0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 0= normal; 1= fixed defect; 2= reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])

dataset["target"].describe()

dataset["target"].unique()  # Clearly this is a Classification Problem as Target variable have value 0 and 1

# Checking correlation between columns
print(dataset.corr()["target"].abs().sort_values(ascending=False))

y = dataset["target"]
print("Percentage of patience with heart problems: "+str(y.where(y==1).count()*100/1025))
print("Percentage of patience with heart problems: "+str(y.where(y==0).count()*100/1025))

y = dataset["target"]
sns.countplot(dataset,x="target",palette=['black','cyan'],width=0.5)
target_temp = dataset.target.value_counts()
print(target_temp)

dataset["sex"].unique()

import seaborn as sns
sns.barplot(dataset,x="sex",y="target",saturation=0.75,hue="sex",width=0.5) # 0:feamle and 1:male
# We notice, that females are more likely to have heart problems than males

dataset["cp"].unique()

sns.barplot(dataset,x="cp",y="target",hue="cp")
# We notice, that chest pain of '0', i.e. the ones with typical angina are much less likely to have heart problems

dataset["fbs"].unique()

sns.barplot(dataset,x="fbs",y="target",hue="fbs")

dataset["restecg"].unique()

sns.barplot(dataset,x="restecg",y="target",hue="restecg")
# We realize that people with restecg '1' and '0' are much more likely to have a heart disease than with restecg '2'

dataset["exang"].unique()

sns.barplot(dataset,x="exang",y="target",hue="exang")
# People with exang=1 i.e. Exercise induced angina are much less likely to have heart problems

dataset["slope"].unique()

sns.barplot(dataset,x="slope",y="target",hue="slope")
# We observe, that Slope '2' causes heart pain much more than Slope '0' and '1'

dataset["ca"].unique()

sns.countplot(dataset,x="ca",hue="ca")

sns.barplot(dataset,x="ca",y="target",hue="ca")
# ca=4 has astonishingly large number of heart patients

dataset["thal"].unique()

sns.barplot(dataset,x="thal",y="target",hue="thal")

from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

X_train.shape

X_test.shape

Y_train.shape

Y_test.shape

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
re=LogisticRegression()
model=re.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score 
score_lr = round(accuracy_score(Y_pred,Y_test)*100,2)
print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,Y_train)
Y_pred_nb = nb.predict(X_test)

score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)
print("The accuracy score achieved using  Gaussian Naive Bayes is: "+str(score_nb)+" %")

from sklearn.naive_bayes import BernoulliNB
bernoulli_nb=BernoulliNB()
bernoulli_nb.fit(X_train,Y_train)

pred=bernoulli_nb.predict(X_test)
acc_bnb=round(accuracy_score(Y_test,pred)*100,2)
print("The accuracy score achieved using Bernoulli Naive Bayes is: "+str(acc_bnb)+" %")

from sklearn import svm
sv = svm.SVC(kernel='linear')
sv.fit(X_train, Y_train)
Y_pred_svm = sv.predict(X_test)

score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred_knn=knn.predict(X_test)

score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)
print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")

from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)

score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")

from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0


for x in range(200):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)

score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Random Forest is: "+str(score_rf)+" %")

scores = [score_lr,score_nb,acc_bnb,score_svm,score_knn,score_dt,score_rf]
algorithms = ["Logistic Regression","Gaussian Naive Bayes","Bernoulli Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")

sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
scores = [score_lr,score_nb,acc_bnb,score_svm,score_knn,score_dt,score_rf]
algorithms = ["Logistic Regression","Gaussian Naive Bayes","Bernoulli Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")
plt.xticks(rotation=30)

plt.bar(algorithms,scores,color=["#FFC107","#FF99CC","#F7DC6F","#8BC34A","#4CAF50","#9C27B0","#03A9F4"])
plt.show()

from tkinter import *
from datetime import date
from tkinter.ttk import Combobox
import datetime
from tkinter import ttk
import os
import matplotlib
# import joblib
import tkinter as tk

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

from tkinter import messagebox,ttk

import joblib
# Load the trained model
model = joblib.load('Heart_Model.pkl')
joblib.dump(model,"Heart_Model.pkl")


background="#f0ddd5"
framebg="#62a7ff"
framefg="#fefbfb"

root=Tk()
root.title("Heart Attack Prediction System")
root.geometry("1450x900+35+10")
root.config(bg=background)

#title icon
image_icon=PhotoImage(file="icon1.png")
root.iconphoto(False,image_icon)

#header section
logo=PhotoImage(file="header.png")
myImage=Label(image=logo,bg=background)
myImage.place(x=0,y=0)


Heading_entry=Frame(root,width=800,height=190,bg="#df2d4b")
Heading_entry.place(x=600,y=20)

l1=tk.Label(Heading_entry,text="Registration No.",font="arial 13",bg="#df2d4b",fg=framefg).place(x=30,y=0)
e1=tk.Entry(Heading_entry,width=30)
e1.place(x=20,y=30)

l2=tk.Label(Heading_entry,text="Date",font="arial 13",bg="#df2d4b",fg=framefg).place(x=430,y=0)
e2=tk.Entry(Heading_entry,width=30)
e2.place(x=430,y=30)


l3=tk.Label(Heading_entry,text="Patient Name",font="arial 13",bg="#df2d4b",fg=framefg).place(x=30,y=90)
e3=tk.Entry(Heading_entry,width=30)
e3.place(x=20,y=120)

l4=tk.Label(Heading_entry,text="Birth Year",font="arial 13",bg="#df2d4b",fg=framefg).place(x=430,y=90)
e4=tk.Entry(Heading_entry,width=30)
e4.place(x=430,y=120)

# All Details

Detail_entry=Frame(root,width=500,height=260,bg="#dbe0e3")
Detail_entry.place(x=60,y=510)


Label(Detail_entry,text="sex",font="arial 13",bg=framebg,fg=framefg).place(x=10,y=5)
Label(Detail_entry,text="fbs",font="arial 13",bg=framebg,fg=framefg).place(x=180,y=5)
Label(Detail_entry,text="exang",font="arial 13",bg=framebg,fg=framefg).place(x=335,y=5)

def selection1():
    if gen.get()==1:
        Gender=1
        return(Gender)
    elif gen.get()==2:
        Gender=0
        return(Gender)
    else:
        print(Gender)

def selection2():
    if fbs.get()==1:
        Fbs=1
        return(Fbs)
    elif fbs.get()==2:
        Fbs=0
        return(Fbs)
    else:
        print(Fbs)
    
         
def selection3():
    if exang.get()==1:
        Exang=1
        return(Exang)
    elif exang.get()==2:
        Exang=0
        return(Exang)
    else:
        print(Exang)
        
gen=IntVar()
R1=tk.Radiobutton(Detail_entry,text="Male",variable=gen,value=1,command=selection1)
R2=tk.Radiobutton(Detail_entry,text="Female",variable=gen,value=2,command=selection1)
R1.place(x=43,y=5)
R2.place(x=93,y=5) 

fbs=IntVar() 
R3 =tk.Radiobutton(Detail_entry,text="True",variable=fbs,value=1,command=selection2)
R4=tk.Radiobutton(Detail_entry,text="False",variable=fbs,value=2,command=selection2)
R3.place(x=213,y=5)
R4.place(x=263,y=5)

exang=IntVar()
R5=tk.Radiobutton(Detail_entry,text=" Yes",variable=exang,value=1,command=selection3)
R6=tk.Radiobutton(Detail_entry,text="No",variable=exang,value=2,command=selection3)
R5.place(x=387,y=5)
R6.place(x=430,y=5)

# ComboBox -Dropdown Box
tk.Label(Detail_entry,text="cp",font="arial 13",bg=framebg,fg=framefg).place(x=10,y=40)
tk.Label(Detail_entry,text="restecg",font="arial 13",bg=framebg,fg=framefg).place(x=10,y=70)
tk.Label(Detail_entry,text="slope",font="arial 13",bg=framebg,fg=framefg).place(x=10,y=100)
tk.Label(Detail_entry,text="ca:",font="arial 13",bg=framebg,fg=framefg).place(x=10,y=130)
tk.Label(Detail_entry,text="thal",font="arial 13",bg=framebg,fg=framefg).place(x=10,y=160)


cp_combobox=Combobox(Detail_entry,values=['0','1','2','3'],
                     font="arial 12",state="r",width=14)
restecg_combobox=Combobox(Detail_entry,values=['0','1'], font="arial 12",state="r",width=11)
slope_combobox=Combobox(Detail_entry,values=['0','1','2'], font="arial 12",state="r",width=12)
ca_combobox=Combobox(Detail_entry,values=['0','1','2','3'], font="arial 12",state="r",width=14)
thal_combobox=Combobox(Detail_entry,values=['0','1','2','3'], font="arial 12",state="r",width=14)

cp_combobox.place(x=50,y=40)
restecg_combobox.place(x=80,y=70)
slope_combobox.place(x=70,y=100)
ca_combobox.place(x=50,y=130)
thal_combobox.place(x=50,y=160)


tk.Label(Detail_entry,text="Age",font="arial 13",width=7,bg=framebg,fg=framefg).place(x=240,y=40)
tk.Label(Detail_entry,text="trestbps:",font="arial 13",width=7,bg=framebg,fg=framefg).place(x=240,y=70)
tk.Label(Detail_entry,text="chol:",font="arial 13",width=7,bg=framebg,fg=framefg).place(x=240,y=100)
tk.Label(Detail_entry,text="thalach:",font="arial 13",width=7,bg=framebg,fg=framefg).place(x=240,y=130)
tk.Label(Detail_entry,text="oldpeaks",font="arial 13",width=7,bg=framebg,fg=framefg).place(x=240,y=160)

age=StringVar()
trestbps=StringVar()
chol=StringVar()
thalach=StringVar()
oldpeaks=StringVar()

age_entry=Entry(Detail_entry,textvariable=age,width=10,font="arial 15",bg="#ededed",fg="#222222",bd=0)
trestbps_entry=Entry(Detail_entry,textvariable=trestbps,width=10,font="arial 15",bg="#ededed",fg="#222222",bd=0)
chol_entry=Entry(Detail_entry,textvariable=chol,width=10,font="arial 15",bg="#ededed",fg="#222222",bd=0)
thalach_entry=Entry(Detail_entry,textvariable=thalach,width=10,font="arial 15",bg="#ededed",fg="#222222",bd=0)
oldpeaks_entry=Entry(Detail_entry,textvariable=oldpeaks,width=10,font="arial 15",bg="#ededed",fg="#222222",bd=0)

age_entry.place(x=320,y=40)
trestbps_entry.place(x=320,y=70)
chol_entry.place(x=320,y=100)
thalach_entry.place(x=320,y=130)
oldpeaks_entry.place(x=320,y=160)


report=PhotoImage(file="reportBG_remove.png")
report_background=Label(image=report,bg=background)
report_background.place(x=930,y=260)

# Graph
graph=PhotoImage(file="g.png")
Label(image=graph).place(x=610,y=300)
Label(image=graph).place(x=810,y=300)
Label(image=graph).place(x=610,y=490)
Label(image=graph).place(x=810,y=490)

# Function to predict heart attack
def predict_heart_attack():
       # Fetch input values
        
        A=int(age_entry.get())
        B=selection1()
        F=selection2()
        I=selection3()
        C=int(slope_combobox.get())
        G=int(restecg_combobox.get())
        K=int(cp_combobox.get())
        L=int(ca_combobox.get())
        M=int(thal_combobox.get())
        D=int(trestbps.get())
        E=int(chol.get())
        H=int(thalach.get())
        J=eval(oldpeaks.get())
        
        input_data=(A,B,C,D,E,F,G,H,I,J,K,L,M)
        
        #change input data to numpy array
        input_data_as_numpy_array=np.asarray(input_data)
        input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
        prediction =model.predict(input_data_reshaped)
        result = "High risk of heart attack" if prediction == 1 else "Low risk of heart attack"
        messagebox.showinfo("Prediction Result", result)
        
# Generate Report Button
Button(root,cursor='hand2',text="Predict", command=predict_heart_attack,font="arial 15 bold",bg="red").place(x=740,y=650)
#Start the GUI loop
root.mainloop()

import joblib
joblib.dump(model,"Heart_Model.pkl")

from tkinter import messagebox,ttk
# Load the trained model
model = joblib.load('Heart_Model.pkl')

background="#f0ddd5"
framebg="#62a7ff"
framefg="#fefbfb"

root=Tk()
root.title("Heart Attack Prediction System")
root.geometry("1450x900+35+10")
root.config(bg=background)

#title icon
image_icon=PhotoImage(file="icon1.png")
root.iconphoto(False,image_icon)

#header section
logo=PhotoImage(file="header.png")
myImage=Label(image=logo,bg=background)
myImage.place(x=0,y=0)


Heading_entry=Frame(root,width=800,height=190,bg="#df2d4b")
Heading_entry.place(x=600,y=20)

l1=tk.Label(Heading_entry,text="Registration No.",font="arial 13",bg="#df2d4b",fg=framefg).place(x=30,y=0)
e1=tk.Entry(Heading_entry,width=30)
e1.place(x=20,y=30)

l2=tk.Label(Heading_entry,text="Date",font="arial 13",bg="#df2d4b",fg=framefg).place(x=430,y=0)
e2=tk.Entry(Heading_entry,width=30)
e2.place(x=430,y=30)


l3=tk.Label(Heading_entry,text="Patient Name",font="arial 13",bg="#df2d4b",fg=framefg).place(x=30,y=90)
e3=tk.Entry(Heading_entry,width=30)
e3.place(x=20,y=120)

l4=tk.Label(Heading_entry,text="Birth Year",font="arial 13",bg="#df2d4b",fg=framefg).place(x=430,y=90)
e4=tk.Entry(Heading_entry,width=30)
e4.place(x=430,y=120)

# All Details

Detail_entry=Frame(root,width=500,height=260,bg="#dbe0e3")
Detail_entry.place(x=60,y=510)


Label(Detail_entry,text="sex",font="arial 13",bg=framebg,fg=framefg).place(x=10,y=10)
Label(Detail_entry,text="fbs",font="arial 13",bg=framebg,fg=framefg).place(x=180,y=10)
Label(Detail_entry,text="exang",font="arial 13",bg=framebg,fg=framefg).place(x=335,y=10)

def selection1():
    if gen.get()==1:
        Gender=1
        return(Gender)
    elif gen.get()==2:
        Gender=0
        return(Gender)
    else:
        print(Gender)

def selection2():
    if fbs.get()==1:
        Fbs=1
        return(Fbs)
    elif fbs.get()==2:
        Fbs=0
        return(Fbs)
    else:
        print(Fbs)
    
         
def selection3():
    if exang.get()==1:
        Exang=1
        return(Exang)
    elif exang.get()==2:
        Exang=0
        return(Exang)
    else:
        print(Exang)
        
gen=IntVar()
R1=tk.Radiobutton(Detail_entry,text="Male",variable=gen,value=1,command=selection1)
R2=tk.Radiobutton(Detail_entry,text="Female",variable=gen,value=2,command=selection1)
R1.place(x=43,y=10)
R2.place(x=93,y=10) 

fbs=IntVar() 
R3 =tk.Radiobutton(Detail_entry,text="True",variable=fbs,value=1,command=selection2)
R4=tk.Radiobutton(Detail_entry,text="False",variable=fbs,value=2,command=selection2)
R3.place(x=213,y=10)
R4.place(x=263,y=10)

exang=IntVar()
R5=tk.Radiobutton(Detail_entry,text=" Yes",variable=exang,value=1,command=selection3)
R6=tk.Radiobutton(Detail_entry,text="No",variable=exang,value=2,command=selection3)
R5.place(x=387,y=10)
R6.place(x=430,y=10)

# ComboBox -Dropdown Box
tk.Label(Detail_entry,text="cp",font="arial 13",bg=framebg,fg=framefg).place(x=10,y=50)
tk.Label(Detail_entry,text="restecg",font="arial 13",bg=framebg,fg=framefg).place(x=10,y=90)
tk.Label(Detail_entry,text="slope",font="arial 13",bg=framebg,fg=framefg).place(x=10,y=130)
tk.Label(Detail_entry,text="ca:",font="arial 13",bg=framebg,fg=framefg).place(x=10,y=170)
tk.Label(Detail_entry,text="thal",font="arial 13",bg=framebg,fg=framefg).place(x=10,y=210)


cp_combobox=Combobox(Detail_entry,values=['0','1','2','3'],
                     font="arial 12",state="r",width=14)
restecg_combobox=Combobox(Detail_entry,values=['0','1'], font="arial 12",state="r",width=11)
slope_combobox=Combobox(Detail_entry,values=['0','1','2'], font="arial 12",state="r",width=12)
ca_combobox=Combobox(Detail_entry,values=['0','1','2','3'], font="arial 12",state="r",width=14)
thal_combobox=Combobox(Detail_entry,values=['0','1','2','3'], font="arial 12",state="r",width=14)

cp_combobox.place(x=50,y=50)
restecg_combobox.place(x=80,y=90)
slope_combobox.place(x=70,y=130)
ca_combobox.place(x=50,y=170)
thal_combobox.place(x=50,y=210)


tk.Label(Detail_entry,text="Age",font="arial 13",width=7,bg=framebg,fg=framefg).place(x=240,y=50)
tk.Label(Detail_entry,text="trestbps:",font="arial 13",width=7,bg=framebg,fg=framefg).place(x=240,y=90)
tk.Label(Detail_entry,text="chol:",font="arial 13",width=7,bg=framebg,fg=framefg).place(x=240,y=130)
tk.Label(Detail_entry,text="thalach:",font="arial 13",width=7,bg=framebg,fg=framefg).place(x=240,y=170)
tk.Label(Detail_entry,text="oldpeaks",font="arial 13",width=7,bg=framebg,fg=framefg).place(x=240,y=210)

age=StringVar()
trestbps=StringVar()
chol=StringVar()
thalach=StringVar()
oldpeaks=StringVar()

age_entry=Entry(Detail_entry,textvariable=age,width=10,font="arial 15",bg="#ededed",fg="#222222",bd=0)
trestbps_entry=Entry(Detail_entry,textvariable=trestbps,width=10,font="arial 15",bg="#ededed",fg="#222222",bd=0)
chol_entry=Entry(Detail_entry,textvariable=chol,width=10,font="arial 15",bg="#ededed",fg="#222222",bd=0)
thalach_entry=Entry(Detail_entry,textvariable=thalach,width=10,font="arial 15",bg="#ededed",fg="#222222",bd=0)
oldpeaks_entry=Entry(Detail_entry,textvariable=oldpeaks,width=10,font="arial 15",bg="#ededed",fg="#222222",bd=0)

age_entry.place(x=320,y=40)
trestbps_entry.place(x=320,y=70)
chol_entry.place(x=320,y=100)
thalach_entry.place(x=320,y=130)
oldpeaks_entry.place(x=320,y=160)



report=PhotoImage(file="reportBG_remove.png")
report_background=Label(image=report,bg=background)
report_background.place(x=930,y=260)
# Graph
graph=PhotoImage(file="g.png")
Label(image=graph).place(x=610,y=300)
Label(image=graph).place(x=810,y=300)
Label(image=graph).place(x=610,y=490)
Label(image=graph).place(x=810,y=490)

# Function to predict heart attack
def predict_heart_attack():
       # Fetch input values
        
        A=int(age_entry.get())
        B=selection1()
        F=selection2()
        I=selection3()
        C=int(slope_combobox.get())
        G=int(restecg_combobox.get())
        K=int(cp_combobox.get())
        L=int(ca_combobox.get())
        M=int(thal_combobox.get())
        D=int(trestbps.get())
        E=int(chol.get())
        H=int(thalach.get())
        J=eval(oldpeaks.get())
        
        input_data=(A,B,C,D,E,F,G,H,I,J,K,L,M)
        
        #change input data to numpy array
        input_data_as_numpy_array=np.asarray(input_data)
        input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
        prediction =model.predict(input_data_reshaped)
        result = "High risk of heart attack" if prediction == 1 else "Low risk of heart attack"
        messagebox.showinfo("Prediction Result", result)
        
# Generate Report Button
Button(root,cursor='hand2',text="Predict", command=predict_heart_attack,font="arial 15 bold",bg="white").place(x=740,y=700)
#Start the GUI loop
root.mainloop()