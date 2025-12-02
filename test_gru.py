import torch
from keypoint_dataset import create_dataloaders
from gru import GRUActionClassifier
from visualization_utils import make_training_plot
from train_generic import clear_checkpoint, restore_checkpoint, save_checkpoint, train_epoch, early_stopping, evaluate_epoch

CHECKPOINT_PATH = "./checkpoints/GRU/"

def main():
    # Data loaders
    tr_loader, val_loader, te_loader = create_dataloaders(batch_size=8)
    # Model
    model = GRUActionClassifier()

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Attempts to restore the latest checkpoint if exists
    print("Loading cnn...")
    model, start_epoch, stats = restore_checkpoint(model, CHECKPOINT_PATH)

    axes = make_training_plot()

    # Evaluate the model
    evaluate_epoch(
        axes,
        tr_loader,
        val_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
        update_plot=False,
        multiclass=True, 
    )


if __name__ == "__main__":
    main()