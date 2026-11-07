class EarlyStopping:

    def __init__(self, patience=10, min_delta=0.001):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss):

        if validation_loss < self.best_loss - self.min_delta:

            self.best_loss = validation_loss
            self.counter = 0

            return True  # Improvement

        else:

            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

            return False