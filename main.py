from src.cooptim import Orchestrator
import logging
import json

logging.basicConfig(
        level=getattr(logging, "INFO", logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
logger = logging.getLogger(__name__)

logger.info("Starting run...")

with open("config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f) or {}

cooptim = Orchestrator(config=config_dict)
solutions = cooptim.run()

logger.info("Run completed !")