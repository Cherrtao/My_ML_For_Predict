"""数据库连接封装。"""
from sqlalchemy import create_engine
from config.dataset_config import POSTGRES_DSN


def get_engine():
    """创建 SQLAlchemy Engine。"""
    return create_engine(POSTGRES_DSN)
