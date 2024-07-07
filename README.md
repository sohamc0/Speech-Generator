# Political Speech Generator
If you are a person who is struggling to come across topics to cover on your next public speech look no further! This LSTM will fill out a sentence given a prompt which will be treated as the initial few words. 

## Usage
Your machine must have the following packages installed...
* ```scipy==1.11.4```, ```tensorflow==2.15.0```, ```gensim==4.3.2```, and ```numpy```\
Go into the src directory and run ```python gen.py```. You will then be prompted to start off your sentence with a few words of your choice.

## Chosen Corpus and Data Cleansing
The transcipts of multiple State of the Union addresses delivered by U.S. presidents in the 21st century were used to train this model. The State of the Union address is a speech directed to the American public once a year.\
The length of the phrases used to train the RNN was a constant. So, a training phrase could comprise a sentence that has been cut off prior to its natural completion or a sentence which starts in the middle rather than the beginning.\
Since there are different topics that a president may mention in a particular year, individual words must meet a threshold for the number of times they appear in the training corpus. Subsequently, an event which was only relevant, for example, in the year 2012 will not be mentioned in a sentence generated by this model. However, an ongoing political situation which was mentioned in multiple speeches across the years will most likely be included in a few generated phrases and sentences.

## Output Examples
```There are too many... -> there are too many troops let it.+ workers. in in for sacrifice brave committed```\
```We must continue to... -> we must continue to manufacturing repair america. coverage. factories experts you. dismantle the to```

## Future Improvements
Although current sentences and phrases produced by the model is unlikely to be considered grammatically correct, they can still be used as a base layout for constructing the following three to four sentences. Nevertheless, one future update to this model would certainly be to ensure that generated sentences are grammatically sound or at least intelligible by the average English-speaking human. This may be implemented by one or a combination of the following...
* further training on a corpus of data
* including a parameter to distinguish between verbs, adjectives, nouns, etc.\
or
* implementing a retrospective lexer and parser to validate sentences