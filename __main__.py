import tensorflow as tf
from tensorflow.keras import layers

import raw_dataset as raw
import vectorize_dataset as vectorize
import train_model
import config
import history_graph


def train():
    # load raw dataset
    raw_train_dataset = raw.load_raw_train_dataset()
    raw_test_dataset = raw.load_raw_test_dataset()
    raw_validation_dataset = raw.load_raw_validation_dataset()

    # vectorize dataset
    vectorize_layer = vectorize.create_text_vectorization_layer(raw_train_dataset)
    train_dataset = vectorize.vectorize_dataset(raw_train_dataset, vectorize_layer)
    test_dataset = vectorize.vectorize_dataset(raw_test_dataset, vectorize_layer)
    validation_dataset = vectorize.vectorize_dataset(raw_validation_dataset, vectorize_layer)

    # build and train model
    trained_model, trained_history = train_model.build_model(train_dataset, validation_dataset)

    # plot training history
    # history_graph.display_training_accuracy_history(trained_history)
    # history_graph.display_training_loss_history(trained_history)

    # evaluate model on test dataset
    loss, accuracy = trained_model.evaluate(test_dataset)
    print("Test accuracy:", accuracy)
    print("Test loss:", loss)

    # save model
    model_saved_path = f"trained_models/sa_model-{config.epochs}epochs-{accuracy:.2f}accuracy-{loss:.2f}loss.keras"
    exported_model = train_model.save_model(vectorize_layer, trained_model, model_saved_path)


def analyze(model_path, sentence):
    # load the model from disk
    loaded_model = train_model.load_model(model_path)
    
    predict_text = tf.constant([])
    if(isinstance(sentence, list)):
        predict_text = tf.constant(sentence)
    else:
        predict_text = tf.constant([sentence])

    prediction = loaded_model.predict(predict_text)
    print("Sentence:\n", sentence)
    print("Prediction: \n", prediction)


def test_samples():
    model_path = "trained_models/sa_model-11epochs-0.85accuracy-0.40loss.keras"

    ## Positive movie review sample from test dataset, predition: 0.9982441, 0.9882496
    # sentence = "My boyfriend and I went to watch The Guardian.At first I didn't want to watch it, but I loved the movie- It was definitely the best movie I have seen in sometime.They portrayed the USCG very well, it really showed me what they do and I think they should really be appreciated more.Not only did it teach but it was a really good movie. The movie shows what the really do and how hard the job is.I think being a USCG would be challenging and very scary. It was a great movie all around. I would suggest this movie for anyone to see.The ending broke my heart but I know why he did it. The storyline was great I give it 2 thumbs up. I cried it was very emotional, I would give it a 20 if I could!"
    # sentence="I went and saw this movie last night after being coaxed to by a few friends of mine. I'll admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy. I was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben Randall with such professionalism. The sign of a good movie is that it can toy with our emotions. This one did exactly that. The entire theater (which was sold out) was overcome by laughter during the first half of the movie, and were moved to tears during the second half. While exiting the theater I not only saw many women in tears, but many full grown men as well, trying desperately not to let anyone see them crying. This movie was great, and I suggest that you go see it before you judge."
    
    ## Negative movie review sample from test dataset, predition: 0.1066813, 0.08647703
    # sentence = "Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific sea rescue sequences, of which there are very few I just did not care about any of the characters. Most of us have ghosts in the closet, and Costner's character are realized early on, and then forgotten until much later, by which time I did not care. The character we should really care about is a very cocky, overconfident Ashton Kutcher. The problem is he comes off as kid who thinks he's better than anyone else around him and shows no signs of a cluttered closet. His only obstacle appears to be winning over Costner. Finally when we are well past the half way point of this stinker, Costner tells us all about Kutcher's ghosts. We are told why Kutcher is driven to be the best with no prior inkling or foreshadowing. No magic here, it was all I could do to keep from turning it off an hour in."
    # sentence = "This is a pale imitation of 'Officer and a Gentleman.' There is NO chemistry between Kutcher and the unknown woman who plays his love interest. The dialog is wooden, the situations hackneyed. It's too long and the climax is anti-climactic(!). I love the USCG, its men and women are fearless and tough. The action scenes are awesome, but this movie doesn't do much for recruiting, I fear. The script is formulaic, but confusing. Kutcher's character is trying to redeem himself for an accident that wasn't his fault? Costner's is raging against the dying of the light, but why? His 'conflict' with his wife is about as deep as a mud puddle. I saw this sneak preview for free and certainly felt I got my money's worth."
    # analyze(model_path, sentence)

    ## Simple positive example, Predictions: Prediction: [[0.97218084][0.9649973 ][0.96529585][0.9660827 ][0.96512604][0.9885002 ][0.9223285 ][0.9747673 ][0.92030174][0.95030713]]
    # positive_sentences = [
    #     "The cinematography in this film is absolutely stunning, capturing each scene with breathtaking beauty.",
    #     "The performances were outstanding, particularly [actor's name], who brought depth and emotion to their character.",
    #     "The storyline is gripping and keeps you on the edge of your seat from start to finish.",
    #     "The soundtrack enhances every moment, perfectly complementing the mood of the film.",
    #     "The special effects were top-notch, creating a truly immersive experience.",
    #     "This movie is a delightful blend of humor and heart, leaving audiences both laughing and moved.",
    #     "The direction is masterful, seamlessly weaving together multiple storylines into a cohesive and engaging narrative.",
    #     "It's refreshing to see a film that tackles important social issues with sensitivity and insight.",
    #     "The production design is meticulous, transporting viewers to another time and place.",
    #     "Overall, this movie is a must-see, offering entertainment and substance in equal measure.",
    # ]
    # analyze(model_path, positive_sentences)

    ## Simple negative example, Predictions: Prediction:[[0.7470509 ][0.837909  ][0.6984268 ][0.7997992 ][0.9105126 ][0.74538654][0.80913955][0.9352385 [0.9009281 ][0.8730284 ]]
    # negative_sentences = [
    #     "The plot was disjointed and confusing, leaving viewers struggling to follow what was happening.",
    #     "The acting felt flat and uninspired, with performances lacking depth and emotion.",
    #     "The dialogue was clichéd and predictable, failing to engage the audience or spark interest.",
    #     "The pacing was painfully slow, making the movie feel much longer than it actually was.",
    #     "The special effects were amateurish and detracted from rather than enhanced the viewing experience.",
    #     "The characters were poorly developed, leaving viewers indifferent to their fates.",
    #     "The cinematography was uninspired, with bland visuals that failed to capture the essence of the story.",
    #     "The soundtrack felt out of place, detracting from rather than enhancing key moments in the film.",
    #     "The directing lacked vision, resulting in a film that felt aimless and lacking in coherence.",
    #     "Overall, this movie fell short of expectations, failing to deliver on its promise and leaving viewers disappointed.",
    # ]
    # analyze(model_path, negative_sentences)

    ## Negative emotion example words over 50 for movie, Predictions: 0.02043398
    # negative_over50_words = [
    #     "The film started with promising potential but quickly unraveled into a tedious and confusing mess. The plot was convoluted, jumping between disconnected storylines without coherence. Characters lacked depth, their motivations unclear and actions inexplicable. Despite a talented cast, performances felt uninspired, failing to salvage a script riddled with clichés and predictable twists. Overall, it was a disappointing experience, leaving viewers bewildered and dissatisfied."
    # ]
    # analyze(model_path, negative_over50_words)

    ## Negative emotion example words over 50 for travel, Predictions: 0.3355006
    negative_over50_words_travel = [
        "My vacation experience was marred by a series of unfortunate events. From delayed flights and lost luggage to subpar accommodations and unhelpful staff, every aspect seemed plagued by mishaps. The tourist attractions were overcrowded, making it impossible to enjoy any peaceful moments. Despite meticulous planning, the trip turned into a stressful ordeal, leaving me wishing I had stayed home instead."
    ]
    analyze(model_path, negative_over50_words_travel)


if __name__ == '__main__':
    ## train model
    # train()

    ## model testing
    test_samples()
