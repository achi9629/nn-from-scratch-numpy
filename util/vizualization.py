import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(config, df, save_path=None):

    epochs = df['Epoch']
    train_loss = df['Loss_train']
    test_loss = df['Loss_test']
    train_acc = df['Accuracy_train']
    test_acc = df['Accuracy_test']

    plt.figure(figsize=(12,5))

    # ---- Loss plot ----
    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, test_loss, label="Test Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss " + config.optimizer.otype)
    plt.legend()
    plt.grid(True)

    # ---- Accuracy plot ----
    plt.subplot(1,2,2)
    plt.plot(epochs, train_acc, label="Train Accuracy", marker="o")
    plt.plot(epochs, test_acc, label="Test Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Test Accuracy "+ config.optimizer.otype)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved plot to {save_path}")


def plot_classification_metrics(config, df, save_path=None):

    gt = df['GT']
    pred = df['Pred']
    # Confusion matrix
    cm = confusion_matrix(gt, pred)
    acc = np.mean(gt == pred)

    # Class names default
    class_names = [str(i) for i in np.unique(gt)]

    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    # ---- Confusion Matrix Heatmap ----
    im = ax[0].imshow(cm, cmap="Blues")
    ax[0].set_title(f"Confusion Matrix (Acc={acc*100:.2f}%) "+ config.optimizer.otype)
    fig.colorbar(im, ax=ax[0])
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Ground Truth")
    ax[0].set_xticks(np.arange(len(class_names)))
    ax[0].set_yticks(np.arange(len(class_names)))
    ax[0].set_xticklabels(class_names, rotation=45)
    ax[0].set_yticklabels(class_names)

    # Annotate each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax[0].text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max()/2 else "black")

    # ---- Per-Class Accuracy ----
    class_acc = cm.diagonal() / cm.sum(axis=1)
    ax[1].bar(class_names, class_acc)
    ax[1].set_ylim([0,1])
    ax[1].set_title("Per-Class Accuracy "+ config.optimizer.otype)
    ax[1].set_xlabel("Class")
    ax[1].set_ylabel("Accuracy")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved metrics plot to {save_path}")
    # plt.show()

    # Print classification report in console
    # print("\nClassification Report:")
    # print(classification_report(gt, pred, target_names=class_names))

def save_plots_line_by_line(img_paths, save_path="combined.png", dpi=180):
    """
    Load multiple PNG plots and save them stacked vertically in a single PNG.
    """
    n = len(img_paths)
    
    # figure height proportional to number of plots
    fig, axes = plt.subplots(n, 1, figsize=(10, 3*n))  
    
    if n == 1:
        axes = [axes]  # ensure iterable
    
    for ax, path in zip(axes, img_paths):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(path, fontsize=9)  # show filename
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Combined image saved to {save_path} (dpi={dpi})")