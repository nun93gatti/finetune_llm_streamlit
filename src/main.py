import logging
from utils.config import load_config
from data.data_loader import DataLoader
from models.model_trainer import ModelTrainer


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config()


if __name__ == "__main__":
    main()
