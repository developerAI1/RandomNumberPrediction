from django.shortcuts import render , redirect
from django.http import HttpResponse,JsonResponse
import numpy as np
from myapp.models import MyArray
import re
import os       
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout,Bidirectional
from random import randint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import model_from_json
import random
from django.views.decorators.csrf import csrf_exempt
from keras.models import model_from_json
from django.db.models import Q
from django.contrib import messages

# with open('/home/mandeep/Downloads/Data based Sequences.txt', 'r') as file:
#     contents = file.read()
# lines = contents.split('\n')
# arrays = []
# i=0
# for i,j in enumerate(lines):
  #    arrays.append(lines[i].split(' ' ))      
#    i= i+1

# numpy_arrays=np.array(arrays)

# def SaveDAtabase(request):
#     for arr in numpy_arrays: create 
#         MyArray.objects.(data=arr)   
#     return HttpResponse(request,'Random arrays saved!')

count_sequence =240

def home(req):
    return render(req , "index.html")


def AIGuess(request):
    file = open('/home/mandeep/Desktop/GitAddAI/AIGuessNumber/saved_model/AIGuessModel.json', 'r')
    loaded  = file.read()
    file.close()
    loaded_model = model_from_json(loaded)
    loaded_model.load_weights("/home/mandeep/Desktop/GitAddAI/AIGuessNumber/saved_model/modelweights.h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_test = np.load('/home/mandeep/Desktop/GitAddAI/AIGuessNumber/saved_model/test_X.npy')
    result=loaded_model.predict(X_test).astype(int)
    result_set=(set(result.flatten())) 
    result_list = [num for num in result_set if num != 0 and num !=26]
    random_values = random.sample(result_list, k=min(3, len(result_list)))
    print(random_values)
    return render(request , "genrate.html" , {"random_values":random_values})
   
  
def prepare_data(request,data):
    num_inputs = 10
    num_outputs = 5
    arrays = []
    
    input_list = []
    output_list = []
    
    # data = MyArray.objects.all().values()
    for list_values in data:
        for key, value in list_values.items():
            if key == 'data':
                arrays.append(value)
    
    for arr in arrays:  
        integer_array = [int(x) for x in re.findall(r'\d+', arr)]
        input_data = np.zeros((len(arrays), num_inputs), dtype=int)
        input_data= integer_array
        input_list.append(input_data)
        arr2=np.array(integer_array)
        # print('------------>',arr2)
        # if len(arr2.tolist()) != 0:
        output_data = np.zeros((len(arrays), num_outputs),dtype=int)
        indices = np.random.choice(10, size=5, replace=False)  
        # print('=============>',indices)
        output_data = arr2[indices]
        # print(output_data)
        output_list.append(output_data)
        # else:
        #     print("List is empty")
            
    input=np.array(input_list)
    output=np.array(output_list)

    # print('Length of data ------------------->>>>>',len(arrays))
    from sklearn.model_selection import train_test_split
    train_x,test_x,train_y,test_y = train_test_split(input,output,test_size=0.2,random_state=42)

    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()

    train_x=scaler.fit_transform(train_x)
    test_x=scaler.transform(test_x)

    train_X=train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
    test_X=test_x.reshape(test_x.shape[0], 1, test_x.shape[1])
    print(train_X.shape)
    print(test_X.shape)
    
    model = Sequential()
    model.add(Bidirectional(LSTM(240,return_sequences=True,input_shape=(1,10))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240,return_sequences=True,input_shape=(1,10))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240,return_sequences=True, input_shape=(1,10))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240,return_sequences=True, input_shape=(1,10))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240,return_sequences=True, input_shape=(1,10))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240,return_sequences=True, input_shape=(1,10))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(240,return_sequences=False, input_shape=(1,10))))
    model.add(Dense(5))

    model.compile(optimizer='adam',loss = 'mse',metrics=['accuracy'])
    model.fit(train_X, train_y, epochs=400, batch_size=100, verbose=1)
    
    model_json = model.to_json()
    with open("/home/mandeep/Desktop/GitAddAI/AIGuessNumber/saved_model/AIGuessModel.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("/home/mandeep/Desktop/GitAddAI/AIGuessNumber/saved_model/modelweights.h5")   

    
@csrf_exempt
def add_new_sequence(request):
    global count_sequence   
    if request.method=='POST':
        NewAddedArray= request.POST.get('array')
        integer_array = [int(x) for x in re.findall(r'\d+', NewAddedArray)]  
        filtered_lst = [item for item in integer_array if item <= 25]
        New=[str(i) for i in filtered_lst if filtered_lst.count(i)==1]
        if len(New)==10:
            latest_id = MyArray.objects.latest('id').id if MyArray.objects.exists() else 0
            arr_obj = MyArray(id=latest_id+1, data=New,status=1)
            arr_obj.save()
            num_status_1 = MyArray.objects.filter(status=1).count()
            if num_status_1 == count_sequence:
                MyArray.objects.filter(status=1).update(status=2)
                data = MyArray.objects.all().values()
                prepare_data(request , data)
            else:
                return redirect("/")
        else:
            # print("!!!!!!!!!!!!!!!!you have entered two times value")
            return redirect("/")
    else:
        return redirect("/")
    
def array_history(request):
    recent_sequences = MyArray.objects.filter(Q(status=1) | Q(status=2))
    print(recent_sequences)
    return render(request, 'history.html' , {'recent_sequences':recent_sequences})  

@csrf_exempt
def Reset_history(request):
    recent_sequences = MyArray.objects.filter(Q(status=1) | Q(status=2))
    val = request.POST.get('input')
    if request.method=='POST':
        val = request.POST.getlist('input')
        if val:
            val=list(map(int,val))
            for i in val:
                MyArray.objects.filter(id=i).delete()
                messages.success(request, "Selected sequences have been deleted.")
            return  redirect('/')
        else:
            messages.warning(request, "No sequences were selected.")
        
    return render(request, 'reset.html' , {'recent_sequences':recent_sequences})  
