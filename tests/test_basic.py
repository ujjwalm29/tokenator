import tempfile
from src.tokenator.base_wrapper import BaseWrapper
from src.tokenator.utils import get_default_db_path


def test_custom_db_path():
    # Create a temporary file to act as the custom db_path
    with tempfile.NamedTemporaryFile() as tmp:
        custom_db_path = tmp.name
        wrapper = BaseWrapper(client=None, db_path=custom_db_path)
        session = wrapper.Session()
        assert session.bind.url.database == custom_db_path
        session.close()


def test_default_db_path():
    # Test with default db_path
    wrapper_default = BaseWrapper(client=None)
    session_default = wrapper_default.Session()
    assert session_default.bind.url.database == get_default_db_path()
    session_default.close()
