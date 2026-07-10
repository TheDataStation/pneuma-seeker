from sys import path

path.append("../../../../../..")

from pneuma_seeker.services.core.api.language_model import LanguageModelAPI
from pneuma_seeker.shared.config import Config
from pneuma_seeker.shared.logger import setup_logger

config = Config("../../../../../../../.env")
logger = setup_logger()
lm_api = LanguageModelAPI(config, logger)

# from pneuma_seeker.services.core.ir_system.retriever.impl.attribute_retriever import index_attribute

# index_attribute("archeology", lm_api)
# index_attribute("astronomy", lm_api)  # 10 minutes
# index_attribute("biomedical", lm_api)
# index_attribute("environment", lm_api)
# index_attribute("legal", lm_api)
# index_attribute("wildfire", lm_api)
