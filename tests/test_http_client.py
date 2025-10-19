import unittest
from unittest.mock import Mock, patch
from coffecrawler.core.http_client import HTTPClient

class TestHTTPClient(unittest.TestCase):
    @patch('coffecrawler.core.http_client.CacheManager')
    def test_get_user_agent_mozilla(self, mock_cache_manager):
        # Mock the crawler instance with the desired configuration
        mock_crawler = Mock()
        mock_crawler.config.agent_module = 'mozilla'
        mock_crawler.mobile_emulation = False

        # Create an instance of HTTPClient with the mocked crawler
        http_client = HTTPClient(mock_crawler)

        # Get the user agent
        user_agent = http_client._get_user_agent()

        # Assert that the user agent is the correct one for desktop Mozilla
        self.assertIn('Mozilla', user_agent)

if __name__ == '__main__':
    unittest.main()
