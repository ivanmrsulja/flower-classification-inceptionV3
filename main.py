import pandas as pd
from sklearn.model_selection import train_test_split

from util import *
from model import *

if __name__ == "__main__":

    df = pd.read_csv("flower_images/flower_images/flower_labels.csv")

    df = df.replace(flower_names)
    
    # Plot the distribution of labeled data
    # df.label.value_counts().plot.bar()

    # Plot a few images
    # img_plot(df)

    # These parameters are determined using experience from other kaggle users
    train_input_shape = (299,299, 3)
    IMG_SIZE = train_input_shape[0]
    SEED = 42

    train_df, validation_df = train_test_split(df, 
                                        test_size=0.2, 
                                        random_state=SEED, 
                                        stratify=df['label'].values)
    
    train_imgs, train_df = create_datasets(train_df, IMG_SIZE)
    validation_imgs, validation_df = create_datasets(validation_df, IMG_SIZE)    

    model = get_compiled_model(image_height=IMG_SIZE, image_width=IMG_SIZE)
    trained_model = train_model(model, train_imgs, train_df, validation_imgs, validation_df)
    plot_loss_and_accuracy(trained_model)
    print_metrics(trained_model, print_accuracy=True, print_f1=True)

    # save_model(model, "my_model.h5")
    test_saved_model("my_model.h5", "demonstration/demo_img_1.jpg", IMG_SIZE)
    