from keypoint_dataset import create_dataloaders
from gru import GRUActionClassifier
from visualization_utils import make_training_plot
from train_generic import clear_checkpoint, restore_checkpoint, save_checkpoint, train_epoch, early_stopping, evaluate_epoch
import matplotlib.pyplot as plt
import torch


CHECKPOINT_PATH = "./checkpoints/GRU/"

def main():
    # Data loaders
    tr_loader, val_loader, te_loader = create_dataloaders(batch_size=8)
    # Model
    model = GRUActionClassifier()
    # define loss function

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # restore checkpoint if exists
    model, start_epoch, stats = restore_checkpoint(model, CHECKPOINT_PATH)
    
    axes = make_training_plot()
 
    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes,
        tr_loader,
        val_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        multiclass=True,
    )

     # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    patience = 50
    curr_count_to_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_count_to_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            val_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            multiclass=True,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, CHECKPOINT_PATH, stats)

        # update early stopping parameters
        curr_count_to_patience, prev_val_loss = early_stopping(stats, curr_count_to_patience, prev_val_loss)

        epoch += 1
    print("Finished Training")
    # Save figure and keep plot open
    plt.savefig(f"GRU_training_plot_patience={patience}.png", dpi=200)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
