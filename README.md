## Introduction


Recurrent Neural Networks(RNN) are a type of Neural Network where the output from the previous step is fed as input to the current step. In traditional neural networks, all the inputs and outputs are independent of each other, but in cases like when it is required to predict the next word of a sentence, the previous words are required and hence there is a need to remember the previous words. Thus RNN came into existence, which solved this issue with the help of a Hidden Layer. The main and most important feature of RNN is Hidden state, which remembers some information about a sequence.
RNN have a “memory” which remembers all information about what has been calculated. It uses the same parameters for each input as it performs the same task on all the inputs or hidden layers to produce the output. This reduces the complexity of parameters, unlike other neural networks.


In our project we have used the RNN to learn the patterns in music that we humans enjoy. Once it learns this, the model should be able to generate new music for us.

## Data Preparation

Our input to the model is a sequence of musical events/notes.In this case-study we have limited ourselves to single instrument music. We have used ABC (ABC notation is a shorthand form of musical notation. In basic form it uses the letters A through G, letter notation, to represent the given notes, with other elements used to place added value on these – sharp, flat, the length of the note, key, ornamentation.)notation in order to represent our input music.

### DATA source:
http://abc.sourceforge.net/NMD/nmd/jigs.txt
http://abc.sourceforge.net/NMD/nmd/hpps.txt


We have used  batch training. This trains the model using only a subsample of data at a time.
We have set following parameters:
Batch Size = 16
Sequence Length = 64

We have found out that there are a total of 155222 characters in our data. Total number of unique characters is 87.
We have assigned a numerical index to each unique character. We have created a dictionary where the key belongs to a character and its value is it’s index. We have also created an opposite of it, where the key belongs to the index and its value is it’s character.
Network Model
We have added an embedding layer as the first hidden layer of a network.
Word embeddings provide a dense representation of words and their relative meanings.In an embedding, words are represented by dense vectors where a vector represents the projection of the word into a continuous vector space

It must specify 3 arguments:
input_dim: This is the size of the vocabulary in the text data.In our case it will be number of unique characters 
output_dim: This is the size of the vector space in which words will be embedded. It defines the size of the output vectors from this layer for each word. 
input_length: This is the length of input sequences, as you would define for any input layer of a Keras model. For example, if all of your input documents are 1000 words, this would be 1000.

We have three such RNN layers each having 256 LSTM units. The output of each LSTM unit will be an input to all of the LSTM units in the next layer and so on. Here, in our project each RNN unit is an LSTM unit
We have used a dropout technique to avoid overfitting at the end of each RNN layer, dropout offers a very computationally cheap and remarkably effective regularization method to reduce overfitting and improve generalization error in deep neural networks. 
After three such layers of RNN, we have applied ‘TimeDistributed’ dense layers with “Softmax” activations in it. This wrapper applies a layer to every temporal slice of an input. Since the shape of each output after third LSTM layer is (16*64*256). We have 87 unique characters in our dataset and we want that the output at each time-stamp will be a next character in the sequence which is one of the 87 characters. So, a time-distributed dense layer contains 87 “Softmax” activations and it creates a dense connection at each time-stamp. Finally, it will generate 87 dimensional output at each time-stamp which will be equivalent to 87 probability values. It helps us to maintain a Many-to-Many relationship.



## Training Model

def training_model(data, epochs = 80):
    #mapping character to index
    char_to_index = {ch: i for (i, ch) in enumerate(sorted(list(set(data))))}
    print("Number of unique characters in our whole tunes database = {}".format(len(char_to_index))) #87
    
    with open(os.path.join(data_directory, charIndex_json), mode = "w") as f:
        json.dump(char_to_index, f)
        
    index_to_char = {i: ch for (ch, i) in char_to_index.items()}
    unique_chars = len(char_to_index)
    
    model = built_model(BATCH_SIZE, SEQ_LENGTH, unique_chars)
    model.summary()
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    
    all_characters = np.asarray([char_to_index[c] for c in data], dtype = np.int32)
    print("Total number of characters = "+str(all_characters.shape[0])) #155222
    
    epoch_number, loss, accuracy = [], [], []
    
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        final_epoch_loss, final_epoch_accuracy = 0, 0
        epoch_number.append(epoch+1)
        
        for i, (x, y) in enumerate(read_batches(all_characters, unique_chars)):
            final_epoch_loss, final_epoch_accuracy = model.train_on_batch(x, y) #check documentation of train_on_batch here: https://keras.io/models/sequential/
            print("Batch: {}, Loss: {}, Accuracy: {}".format(i+1, final_epoch_loss, final_epoch_accuracy))
            #here, above we are reading the batches one-by-one and train our model on each batch one-by-one.
        loss.append(final_epoch_loss)
        accuracy.append(final_epoch_accuracy)
        
        #saving weights after every 10 epochs
        if (epoch + 1) % 10 == 0:
            if not os.path.exists(model_weights_directory):
                os.makedirs(model_weights_directory)
            model.save_weights(os.path.join(model_weights_directory, "Weights_{}.h5".format(epoch+1)))
            print('Saved Weights at epoch {} to file Weights_{}.h5'.format(epoch+1, epoch+1))
    
    #creating dataframe and record all the losses and accuracies at each epoch
    log_frame = pd.DataFrame(columns = ["Epoch", "Loss", "Accuracy"])
    log_frame["Epoch"] = epoch_number
    log_frame["Loss"] = loss
    log_frame["Accuracy"] = accuracy
    log_frame.to_csv("../Data/log.csv", index = False)

```bash
Character index : 50
Length        : 100
Output
```
"Em"e3 a3|"A"g/ga3 eaa|"D"ffa a2g|"G"gfg dcB|
"A7"BAA ABA|"B7"B2A EGF|"E7
```

Eg 2
Character index : 40
Length        : 100
Output
```
"G"(3GABd BdBG|"G"[B3B_2B B3|"G"dcB

Eg 3
```
Character index : 10
Length        : 500
```
"G"g2d "Bm"d2B|"Bm"A2B "E7"e2d|"Em"B2B Bcd|"Am"c3 "D7"A3|"G"G2F B2d|"G"g3 f2e|
"G"d2B B^=cB|"A7"AGA "D7"FGA|"G"B^GD D3|
"G"G2G BGG|"G"GBg "C"g3|"C"e3 "D7"ddB|"C"G2E EFG|"D7"FGA "G"G3||
"G"ded BdB|(3"G"GBd "C"efg|"D"f3 "D7"d3|"G"dBG "C"E2G|\
"G"dBG GBd|"C"c3 c2:|

We have used a online music converter that converts abc notation to mp3
https://melobytes.com/en/app/melobytes

The generated music are available here
https://drive.google.com/open?id=1ALLvCLMLFE11UoRgNlOmvAECQvVPnTGv

Full source code available at
https://github.com/Arjunumesh11/music_generation

