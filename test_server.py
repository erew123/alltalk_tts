import asyncio
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import aiohttp

# pylint: disable=all

class AllTalkTester:
    def __init__(self):
        self.banner = """     
             _    _ _ _____     _ _       _____ _____ ____ 
            / \  | | |_   _|_ _| | | __  |_   _|_   _/ ___|
           / _ \ | | | | |/ _` | | |/ /    | |   | | \___ \ 
          / ___ \| | | | | (_| | |   <     | |   | |  ___) |
         /_/   \_\_|_| |_|\__,_|_|_|\_\    |_|   |_| |____/ 
        
               ╔════════════════════════════════════╗
               ║     Server Testing Suite v1.1      ║
               ╚════════════════════════════════════╝
        """
        self.config = {
            "host": "127.0.0.1",
            "port": 7851,
            "log_file": "alltalk_test_results.log",
            "timeout": 30,  # Default timeout for tests
            "retry_count": 3,  # Number of retries for failed tests
            "retry_delay": 5,  # Delay between retries in seconds
        }
        self.narrator_test_cases = {
            "narrator_simple": {
                "text": "*The old wizard gazed at the stars.* \"What mysteries do they hold?\" *he wondered aloud.*",
                "expect_files": 3
            },
            "mixed_complex": {
                "text": "*The wind howled through the trees.* \"I don't like this place,\" *Sarah whispered.* \"Neither do I,\" *John replied, his voice trembling.*",
                "expect_files": 5
            },
            "silent_test": {
                "text": "*Silent narrator text.* \"Speaking character.\" Plain text here.",
                "narrator_enabled": "silent",
                "text_not_inside": "silent",
                "expect_files": 1
            }         
        }
        
        self.filtering_test_cases = {
            "text": "Test with special chars: @#$% and special é character and HTML tags <b>bold</b>",
            "filters": ["none", "standard", "html"]
        }
        
        self.volume_test_cases = [
            {"volume": 0.5, "should_succeed": True},
            {"volume": 1.5, "should_succeed": False}
        ]
        
        self.error_test_cases = [
            {"case": "invalid_voice", "voice": "nonexistent.wav", "expect_error": 500},
            {"case": "bad_language", "language": "xx", "expect_error": 400},
            {"case": "missing_text", "text": "", "expect_error": 422}
        ]        
        self.test_results = []
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging to both file and console"""
        self.log_file = datetime.now().strftime('alltalk_test_%Y%m%d_%H%M%S.log')
        
        # Configure logging with UTF-8 encoding
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("AllTalk Testing Suite Started")
        
    async def setup(self):
        """Initial setup and configuration"""
        print(self.banner)
        self.logger.info("Starting test suite setup")
        
        try:
            self.host = input(f"Enter server IP/URL [{self.config['host']}]: ") or self.config['host']
            self.port = input(f"Enter port [{self.config['port']}]: ") or self.config['port']
            self.base_url = f"http://{self.host}:{self.port}"
            
            self.logger.info(f"Configuration set - Host: {self.host}, Port: {self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            return False

    async def menu(self):
        """Display and handle test selection menu"""
        while True:
            print("\nTest Options:")
            print("1. Run All Tests (takes 30 seconds)")
            print("2. Network Connectivity Tests")
            print("3. Core API Tests")
            print("4. TTS Generation Tests")
            print("5. Error Case Tests")
            print("6. Individual Test Selection")
            print("7. Exit")
            
            try:
                choice = input("\nSelect option: ")
                
                if choice == "1":
                    await self.run_all_tests()
                elif choice == "2":
                    await self.run_network_tests()
                elif choice == "3":
                    await self.run_core_api_tests()
                elif choice == "4":
                    await self.run_tts_tests()
                elif choice == "5":
                    await self._test_error_cases()
                elif choice == "6":
                    await self.individual_test_menu()
                elif choice == "7":
                    self.logger.info("Test suite shutting down")
                    break
                else:
                    print("Invalid option. Please try again.")
                    
            except Exception as e:
                self.logger.error(f"Menu operation failed: {str(e)}")

    async def individual_test_menu(self):
        """Sub-menu for selecting individual tests"""
        while True:
            print("\nSelect Individual Test:")
            print("1. Text Filtering Tests")
            print("2. Narrator Mode Tests")
            print("3. Autoplay Tests")
            print("4. Additional Language Test")
            print("5. OpenAI Compatibility Tests")
            print("6. Return to Main Menu")
            
            try:
                choice = input("\nSelect option: ")
                
                if choice == "1":
                    await self._test_text_filtering()
                elif choice == "2":
                    await self._test_narrator_modes()
                elif choice == "3":
                    await self._test_autoplay()
                elif choice == "4":
                    await self._test_additional_language()
                elif choice == "5":
                    await self._test_openai_tts(self._generate_test_texts()["openai"])
                elif choice == "6":
                    break
                else:
                    print("Invalid option. Please try again.")
                    
            except Exception as e:
                self.logger.error(f"Individual test selection failed: {str(e)}")

    def log_result(self, category: str, test_name: str, status: str, details: str = ""):
        """Log a test result to both the results list and logger"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "test": test_name,
            "status": status,
            "details": details
        }
        self.test_results.append(result)
        
        if status == "PASS":
            self.logger.info(f"{category} - {test_name}: {status}")
        elif status == "WARNING":
            self.logger.warning(f"{category} - {test_name}: {status} - {details}")
        else:
            self.logger.error(f"{category} - {test_name}: {status} - {details}")

    async def run_network_tests(self):
        """Perform network connectivity tests"""
        self.logger.info("Starting network connectivity tests")
        
        # Ping test
        await self._run_ping_test()
        
        # Basic connectivity test
        await self._test_server_connection()
        
    async def _run_ping_test(self):
        """Perform ping test to server"""
        try:
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            command = ['ping', param, '4', self.host]
            
            self.logger.info(f"Running ping test to {self.host}")
            output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode()
            
            if "unreachable" in output.lower() or "timed out" in output.lower():
                self.log_result("Network", "Ping Test", "WARNING", 
                              "Host reachable but some packets lost")
            else:
                self.log_result("Network", "Ping Test", "PASS", 
                              "Host responding to ping")
                
        except subprocess.CalledProcessError as e:
            self.log_result("Network", "Ping Test", "WARNING",
                          f"Ping failed but continuing with tests: {str(e)}")
        except Exception as e:
            self.log_result("Network", "Ping Test", "ERROR",
                          f"Unexpected error during ping test: {str(e)}")

    async def _test_server_connection(self):
        """Test basic server connectivity using /api/ready endpoint"""
        self.logger.info("Testing server API connectivity")
        
        async with aiohttp.ClientSession() as session:
            try:
                start_time = time.time()
                retries = 0
                
                while retries < self.config['retry_count']:
                    try:
                        async with session.get(
                            f"{self.base_url}/api/ready",
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            
                            if response.status == 200:
                                content = await response.text()
                                if content == "Ready":
                                    elapsed = time.time() - start_time
                                    self.log_result(
                                        "Network", 
                                        "Server Connection", 
                                        "PASS",
                                        f"Server ready. Response time: {elapsed:.2f}s"
                                    )
                                    return True
                                else:
                                    self.log_result(
                                        "Network",
                                        "Server Connection",
                                        "WARNING",
                                        f"Server responded but not ready: {content}"
                                    )
                            else:
                                self.log_result(
                                    "Network",
                                    "Server Connection",
                                    "WARNING",
                                    f"Unexpected status code: {response.status}"
                                )
                                
                    except aiohttp.ClientError as e:
                        self.logger.warning(f"Connection attempt {retries + 1} failed: {str(e)}")
                        
                    retries += 1
                    if retries < self.config['retry_count']:
                        await asyncio.sleep(self.config['retry_delay'])
                
                self.log_result(
                    "Network",
                    "Server Connection",
                    "ERROR",
                    f"Server not responding after {retries} attempts"
                )
                return False
                
            except Exception as e:
                self.log_result(
                    "Network",
                    "Server Connection",
                    "ERROR",
                    f"Unexpected error testing server connection: {str(e)}"
                )
                return False

    async def run_core_api_tests(self):
        """Run tests for core API functionality"""
        self.logger.info("Starting core API tests")
        
        # Test core endpoints
        settings = await self._test_current_settings()
        if not settings:
            self.logger.error("Failed to get current settings, skipping dependent tests")
            return
            
        await self._test_voices()
        await self._test_rvc_voices()
        await self._test_root_endpoint()
        
    async def _test_current_settings(self) -> Optional[dict]:
        """Test /api/currentsettings endpoint"""
        self.logger.info("Testing current settings endpoint")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/api/currentsettings",
                    timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                ) as response:
                    if response.status == 200:
                        settings = await response.json()
                        
                        # Validate essential settings exist
                        required_keys = [
                            'current_engine_loaded',
                            'current_model_loaded',
                            'streaming_capable',
                            'multivoice_capable'
                        ]
                        
                        missing_keys = [key for key in required_keys if key not in settings]
                        
                        if missing_keys:
                            self.log_result(
                                "Core API",
                                "Current Settings",
                                "WARNING",
                                f"Missing required settings: {', '.join(missing_keys)}"
                            )
                        else:
                            self.log_result(
                                "Core API",
                                "Current Settings",
                                "PASS",
                                f"Engine: {settings['current_engine_loaded']}, Model: {settings['current_model_loaded']}"
                            )
                            
                        return settings
                    else:
                        self.log_result(
                            "Core API",
                            "Current Settings",
                            "ERROR",
                            f"Unexpected status code: {response.status}"
                        )
                        return None
                        
            except Exception as e:
                self.log_result(
                    "Core API",
                    "Current Settings",
                    "ERROR",
                    f"Failed to get current settings: {str(e)}"
                )
                return None
                
    async def _test_voices(self):
        """Test /api/voices endpoint"""
        self.logger.info("Testing voices endpoint")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/api/voices",
                    timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        voices = data.get('voices', [])
                        
                        if voices:
                            self.log_result(
                                "Core API",
                                "Voices",
                                "PASS",
                                f"Found {len(voices)} voices"
                            )
                            # Store voices for later use in TTS tests
                            self.available_voices = voices
                        else:
                            self.log_result(
                                "Core API",
                                "Voices",
                                "WARNING",
                                "No voices found"
                            )
                    else:
                        self.log_result(
                            "Core API",
                            "Voices",
                            "ERROR",
                            f"Unexpected status code: {response.status}"
                        )
                        
            except Exception as e:
                self.log_result(
                    "Core API",
                    "Voices",
                    "ERROR",
                    f"Failed to get voices: {str(e)}"
                )
                
    async def _test_rvc_voices(self):
        """Test /api/rvcvoices endpoint"""
        self.logger.info("Testing RVC voices endpoint")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/api/rvcvoices",
                    timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        rvc_voices = data.get('rvcvoices', [])
                        
                        if "Disabled" in rvc_voices and len(rvc_voices) == 1:
                            self.log_result(
                                "Core API",
                                "RVC Voices",
                                "PASS",
                                "RVC is disabled (expected behavior)"
                            )
                        elif rvc_voices:
                            self.log_result(
                                "Core API",
                                "RVC Voices",
                                "PASS",
                                f"Found {len(rvc_voices)} RVC voices"
                            )
                            # Store RVC voices for later use
                            self.available_rvc_voices = rvc_voices
                        else:
                            self.log_result(
                                "Core API",
                                "RVC Voices",
                                "WARNING",
                                "No RVC voices found"
                            )
                    else:
                        self.log_result(
                            "Core API",
                            "RVC Voices",
                            "ERROR",
                            f"Unexpected status code: {response.status}"
                        )
                        
            except Exception as e:
                self.log_result(
                    "Core API",
                    "RVC Voices",
                    "ERROR",
                    f"Failed to get RVC voices: {str(e)}"
                )
                
    async def _test_root_endpoint(self):
        """Test root endpoint /"""
        self.logger.info("Testing root endpoint")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    self.base_url,
                    timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        if content:
                            self.log_result(
                                "Core API",
                                "Root Endpoint",
                                "PASS",
                                "Root endpoint returning content"
                            )
                        else:
                            self.log_result(
                                "Core API",
                                "Root Endpoint",
                                "WARNING",
                                "Root endpoint returned empty content"
                            )
                    else:
                        self.log_result(
                            "Core API",
                            "Root Endpoint",
                            "ERROR",
                            f"Unexpected status code: {response.status}"
                        )
                        
            except Exception as e:
                self.log_result(
                    "Core API",
                    "Root Endpoint",
                    "ERROR",
                    f"Failed to access root endpoint: {str(e)}"
                )

    def _generate_test_texts(self) -> dict:
        """Generate various test texts for different TTS scenarios"""
        return {
            "simple": "This is a simple test of the TTS system.",
            "narrator": """*The narrator speaks first.* "Then a character speaks," *and the narrator returns.*""",
            "mixed": """*It was a dark and stormy night.* "What's that noise?" *asked Sarah, her voice trembling.* "Just the wind," *replied John confidently.*""",
            "openai": "Testing the OpenAI compatible endpoint with this text.",
            "long": "This is a longer text that will be used to test the streaming capabilities of the TTS system. It includes multiple sentences to ensure we have enough content to properly test the streaming functionality."
        }

    async def _select_test_voice(self) -> Optional[str]:
        """Select a random voice from available voices"""
        if not hasattr(self, 'available_voices') or not self.available_voices:
            self.log_result(
                "TTS",
                "Voice Selection",
                "ERROR",
                "No voices available for testing"
            )
            return None
        return self.available_voices[0]  # Use first available voice for consistency

    async def run_tts_tests(self):
        """Run comprehensive TTS generation tests"""
        self.logger.info("Starting TTS generation tests")
        
        test_voice = await self._select_test_voice()
        if not test_voice:
            return
            
        # Standard TTS test with text filtering variations
        await self._test_text_filtering()
        
        # Narrator mode tests - now checks for available voices
        voices = await self._select_two_voices()
        if voices:
            await self._test_narrator_modes()
        
        # Autoplay tests
        await self._test_autoplay()
        
        # Additional language test
        await self._test_additional_language()
        
        # Error case testing
        await self._test_error_cases()
        
        # OpenAI compatibility tests
        await self._test_openai_tts(self._generate_test_texts()["openai"])

    async def _test_standard_tts(self, text: str, voice: str):
        """Test standard TTS generation"""
        self.logger.info("Testing standard TTS generation")
        
        async with aiohttp.ClientSession() as session:
            try:
                data = {
                    "text_input": text,
                    "character_voice_gen": voice,
                    "language": "en",
                    "output_file_name": f"test_standard_{int(time.time())}",
                    "text_filtering": "standard"
                }
                
                async with session.post(
                    f"{self.base_url}/api/tts-generate",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("status") == "generate-success":
                            self.log_result(
                                "TTS",
                                "Standard Generation",
                                "PASS",
                                f"Output file: {result.get('output_file_path')}"
                            )
                            # Store successful generation for later tests
                            self.last_generated_file = result.get('output_file_path')
                        else:
                            self.log_result(
                                "TTS",
                                "Standard Generation",
                                "ERROR",
                                f"Generation failed: {result}"
                            )
                    else:
                        self.log_result(
                            "TTS",
                            "Standard Generation",
                            "ERROR",
                            f"Unexpected status code: {response.status}"
                        )
                        
            except Exception as e:
                self.log_result(
                    "TTS",
                    "Standard Generation",
                    "ERROR",
                    f"Failed to generate TTS: {str(e)}"
                )

    async def _test_narrator_tts(self, text: str, voice: str):
        """Test narrator mode TTS generation"""
        self.logger.info("Testing narrator mode TTS generation")
        
        async with aiohttp.ClientSession() as session:
            try:
                data = {
                    "text_input": text,
                    "character_voice_gen": voice,
                    "narrator_voice_gen": voice,  # Using same voice for simplicity
                    "narrator_enabled": "true",
                    "language": "en",
                    "output_file_name": f"test_narrator_{int(time.time())}",
                    "text_filtering": "standard",
                    "text_not_inside": "narrator"
                }
                
                async with session.post(
                    f"{self.base_url}/api/tts-generate",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("status") == "generate-success":
                            self.log_result(
                                "TTS",
                                "Narrator Mode",
                                "PASS",
                                f"Output file: {result.get('output_file_path')}"
                            )
                            # Store successful generation for later tests
                            self.last_narrator_file = result.get('output_file_path')
                        else:
                            self.log_result(
                                "TTS",
                                "Narrator Mode",
                                "ERROR",
                                f"Generation failed: {result}"
                            )
                    else:
                        self.log_result(
                            "TTS",
                            "Narrator Mode",
                            "ERROR",
                            f"Unexpected status code: {response.status}"
                        )
                        
            except Exception as e:
                self.log_result(
                    "TTS",
                    "Narrator Mode",
                    "ERROR",
                    f"Failed to generate narrator TTS: {str(e)}"
                )
           
    async def _test_streaming_tts(self, text: str, voice: str):
        """Test streaming TTS generation"""
        self.logger.info("Testing streaming TTS generation")
        
        async with aiohttp.ClientSession() as session:
            try:
                params = {
                    "text": text,
                    "voice": voice,
                    "language": "en",
                    "output_file": f"test_streaming_{int(time.time())}"
                }
                
                async with session.get(
                    f"{self.base_url}/api/tts-generate-streaming",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                ) as response:
                    if response.status == 200:
                        # Test streaming by reading chunks
                        chunks_received = 0
                        async for chunk in response.content.iter_any():
                            chunks_received += 1
                            
                        if chunks_received > 0:
                            self.log_result(
                                "TTS",
                                "Streaming Generation",
                                "PASS",
                                f"Received {chunks_received} chunks"
                            )
                        else:
                            self.log_result(
                                "TTS",
                                "Streaming Generation",
                                "WARNING",
                                "Stream succeeded but no chunks received"
                            )
                    elif response.status == 400:
                        error_data = await response.json()
                        if "streaming not supported" in error_data.get("error", "").lower():
                            self.log_result(
                                "TTS",
                                "Streaming Generation",
                                "PASS",
                                "Correctly reported streaming not supported"
                            )
                        else:
                            self.log_result(
                                "TTS",
                                "Streaming Generation",
                                "ERROR",
                                f"Bad request: {error_data}"
                            )
                    else:
                        self.log_result(
                            "TTS",
                            "Streaming Generation",
                            "ERROR",
                            f"Unexpected status code: {response.status}"
                        )
                        
            except Exception as e:
                self.log_result(
                    "TTS",
                    "Streaming Generation",
                    "ERROR",
                    f"Failed to test streaming: {str(e)}"
                )

    async def _test_openai_tts(self, text: str):
        """Test OpenAI compatible endpoint"""
        self.logger.info("Testing OpenAI compatible endpoint")
        self.logger.info(f"Testing OpenAI generation with text: {text}")
        
        test_voices = ["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
        test_formats = ["wav", "mp3", "opus", "aac"]
        
        for voice in test_voices[:2]:  # Test first two voices only
            for format in test_formats[:2]:  # Test first two formats only
                await self._test_openai_single_case(text, voice, format)

    async def _test_openai_single_case(self, text: str, voice: str, format: str):
        """Test single case for OpenAI endpoint"""
        self.logger.info(f"Testing OpenAI voice: {voice}, format: {format}")
        self.logger.info(f"Testing OpenAI generation with text: {text}")
        
        payload = {
            "model": "tts-1",  # Model name doesn't matter
            "input": text,
            "voice": voice,
            "response_format": format,
            "speed": 1.0
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/v1/audio/speech",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                ) as response:
                    if response.status == 200:
                        # Check if we received audio data
                        content_type = response.headers.get('Content-Type', '')
                        if content_type.startswith('audio/'):
                            data = await response.read()
                            if len(data) > 0:
                                self.log_result(
                                    "TTS",
                                    f"OpenAI Generation ({voice}, {format})",
                                    "PASS",
                                    f"Generated audio size: {len(data)} bytes"
                                )
                            else:
                                self.log_result(
                                    "TTS",
                                    f"OpenAI Generation ({voice}, {format})",
                                    "WARNING",
                                    "Response successful but no audio data"
                                )
                        else:
                            self.log_result(
                                "TTS",
                                f"OpenAI Generation ({voice}, {format})",
                                "ERROR",
                                f"Unexpected content type: {content_type}"
                            )
                    else:
                        error_data = await response.json()
                        self.log_result(
                            "TTS",
                            f"OpenAI Generation ({voice}, {format})",
                            "ERROR",
                            f"Failed with status {response.status}: {error_data}"
                        )
                        
            except Exception as e:
                self.log_result(
                    "TTS",
                    f"OpenAI Generation ({voice}, {format})",
                    "ERROR",
                    f"Failed to test OpenAI generation: {str(e)}"
                )

    async def _print_test_summary(self):
        """Print summary of all test results"""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['status'] == 'PASS')
        warnings = sum(1 for r in self.test_results if r['status'] == 'WARNING')
        errors = sum(1 for r in self.test_results if r['status'] == 'ERROR')
        
        summary = f"""
        Test Summary
        ============
        Total Tests: {total}
        Passed: {passed}
        Warnings: {warnings}
        Errors: {errors}
        """
        
        print(summary)
        self.logger.info(summary)                

    async def _select_two_voices(self) -> Optional[Tuple[str, str]]:
        """Select two different voices for narrator testing."""
        if not hasattr(self, 'available_voices') or len(self.available_voices) < 2:
            self.log_result(
                "TTS",
                "Voice Selection",
                "SKIP",
                "Need at least 2 voices for narrator testing"
            )
            return None
        return self.available_voices[0], self.available_voices[1]

    async def _test_narrator_modes(self):
        """Test various narrator mode combinations"""
        self.logger.info("Testing narrator mode combinations")
        
        voices = await self._select_two_voices()
        if not voices:
            return
            
        character_voice, narrator_voice = voices
        self.logger.info(f"Using character voice: {character_voice}, narrator voice: {narrator_voice}")
        
        for test_name, case in self.narrator_test_cases.items():
            self.logger.info(f"Running narrator test case: {test_name}")
            
            data = {
                "text_input": case["text"],
                "character_voice_gen": character_voice,
                "narrator_voice_gen": narrator_voice,
                "narrator_enabled": case.get("narrator_enabled", "true"),
                "text_not_inside": case.get("text_not_inside", "narrator"),
                "language": "en",
                "output_file_name": f"test_narrator_{test_name}_{int(time.time())}",
                "output_file_timestamp": True
            }
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        f"{self.base_url}/api/tts-generate",
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            if result.get("status") == "generate-success":
                                self.log_result(
                                    "TTS",
                                    f"Narrator Test {test_name}",
                                    "PASS",
                                    f"Output file: {result.get('output_file_path')}"
                                )
                            else:
                                self.log_result(
                                    "TTS",
                                    f"Narrator Test {test_name}",
                                    "ERROR",
                                    f"Generation failed: {result}"
                                )
                except Exception as e:
                    self.log_result(
                        "TTS",
                        f"Narrator Test {test_name}",
                        "ERROR",
                        f"Test failed: {str(e)}"
                    )

    async def _test_text_filtering(self):
        """Test different text filtering options"""
        self.logger.info("Testing text filtering options")
        
        test_voice = await self._select_test_voice()
        if not test_voice:
            return
            
        test_text = self.filtering_test_cases["text"]
        
        for filter_type in self.filtering_test_cases["filters"]:
            self.logger.info(f"Testing filter type: {filter_type}")
            self.logger.info(f"Testing with text: {test_text}")
            
            data = {
                "text_input": test_text,
                "character_voice_gen": test_voice,
                "text_filtering": filter_type,
                "language": "en",
                "output_file_name": f"test_filter_{filter_type}_{int(time.time())}",
                "output_file_timestamp": True
            }
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        f"{self.base_url}/api/tts-generate",
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            if result.get("status") == "generate-success":
                                self.log_result(
                                    "TTS",
                                    f"Text Filter {filter_type}",
                                    "PASS",
                                    f"Generated with {filter_type} filtering"
                                )
                            else:
                                self.log_result(
                                    "TTS",
                                    f"Text Filter {filter_type}",
                                    "ERROR",
                                    f"Generation failed: {result}"
                                )
                except Exception as e:
                    self.log_result(
                        "TTS",
                        f"Text Filter {filter_type}",
                        "ERROR",
                        f"Test failed: {str(e)}"
                    )

    async def _test_autoplay(self):
        """Test autoplay functionality with different volumes"""
        self.logger.info("Testing autoplay functionality")
        
        try:
            test_voice = await self._select_test_voice()
            if not test_voice:
                return
                
            for test_case in self.volume_test_cases:
                try:
                    volume = test_case["volume"]
                    self.logger.info(f"Testing autoplay with volume: {volume}")

                    data = {
                        "text_input": "Testing autoplay functionality with different volume levels.",
                        "character_voice_gen": test_voice,
                        "language": "en",
                        "output_file_name": f"test_autoplay_{int(time.time())}",
                        "autoplay": True,
                        "autoplay_volume": volume
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        try:
                            async with session.post(
                                f"{self.base_url}/api/tts-generate",
                                data=data,
                                timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                            ) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    expected_success = test_case["should_succeed"]
                                    
                                    if result.get("status") == "generate-success" and expected_success:
                                        self.log_result(
                                            "TTS",
                                            f"Autoplay Volume {volume}",
                                            "PASS",
                                            "Audio should be playing"
                                        )
                                    elif result.get("status") != "generate-success" and not expected_success:
                                        self.log_result(
                                            "TTS",
                                            f"Autoplay Volume {volume}",
                                            "PASS",
                                            "Correctly rejected invalid volume"
                                        )
                                    else:
                                        self.log_result(
                                            "TTS",
                                            f"Autoplay Volume {volume}",
                                            "ERROR",
                                            f"Unexpected result: {result}"
                                        )
                        except Exception as e:
                            self.log_result(
                                "TTS",
                                f"Autoplay Volume {volume}",
                                "ERROR",
                                f"Test failed: {str(e)}"
                            )
                            
                except asyncio.CancelledError:
                    self.logger.warning("Autoplay test cancelled - cleaning up")
                    raise
                except Exception as e:
                    self.log_result(
                        "TTS",
                        f"Autoplay Volume {volume}",
                        "ERROR",
                        f"Test failed: {str(e)}"
                    )
                    
        except asyncio.CancelledError:
            self.logger.warning("Autoplay tests cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Autoplay testing failed: {str(e)}")                            

    async def _test_error_cases(self):
        """Test various error cases to ensure proper error handling"""
        self.logger.info("Testing error cases")
        
        test_voice = await self._select_test_voice()
        if not test_voice:
            return
            
        for case in self.error_test_cases:
            self.logger.info(f"Testing error case: {case['case']}")
            
            # Build base data with known good values
            data = {
                "text_input": "Test text",
                "character_voice_gen": test_voice,
                "language": "en",
                "output_file_name": f"test_error_{int(time.time())}"
            }
            
            # Modify data based on error case
            if case['case'] == "invalid_voice":
                data["character_voice_gen"] = case["voice"]
            elif case['case'] == "bad_language":
                data["language"] = case["language"]
            elif case['case'] == "missing_text":
                data["text_input"] = case["text"]
                
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        f"{self.base_url}/api/tts-generate",
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                    ) as response:
                        response_data = await response.json()
                        
                        if response.status == case["expect_error"]:
                            # Check if error message is user-friendly
                            error_msg = response_data.get("error", "")
                            if error_msg and len(error_msg) < 200 and not any(x in error_msg.lower() for x in ["traceback", "exception", "stack"]):
                                self.log_result(
                                    "Error Cases",
                                    f"Error Test {case['case']}",
                                    "PASS",
                                    f"Received expected error: {error_msg}"
                                )
                            else:
                                self.log_result(
                                    "Error Cases",
                                    f"Error Test {case['case']}",
                                    "WARNING",
                                    "Error received but message might not be user-friendly"
                                )
                        else:
                            self.log_result(
                                "Error Cases",
                                f"Error Test {case['case']}",
                                "ERROR",
                                f"Expected status {case['expect_error']}, got {response.status}"
                            )
                except Exception as e:
                    self.log_result(
                        "Error Cases",
                        f"Error Test {case['case']}",
                        "ERROR",
                        f"Test failed: {str(e)}"
                    )

    async def _test_additional_language(self):
        """Test TTS generation with an additional language if supported"""
        self.logger.info("Testing additional language support")
        
        # Check if multiple languages are supported
        if not hasattr(self, 'current_settings') or not self.current_settings.get('languages_capable'):
            self.log_result(
                "TTS",
                "Additional Language",
                "SKIP",
                "Multiple languages not supported by current engine"
            )
            return
            
        test_voice = await self._select_test_voice()
        if not test_voice:
            return
            
        test_language = "es"  # Spanish as test language
        test_text = "Prueba de texto en español"
        
        data = {
            "text_input": test_text,
            "character_voice_gen": test_voice,
            "language": test_language,
            "output_file_name": f"test_lang_{int(time.time())}",
            "output_file_timestamp": True
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/tts-generate",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("status") == "generate-success":
                            self.log_result(
                                "TTS",
                                "Additional Language",
                                "PASS",
                                f"Generated {test_language} audio"
                            )
                        else:
                            self.log_result(
                                "TTS",
                                "Additional Language",
                                "ERROR",
                                f"Generation failed: {result}"
                            )
                    else:
                        self.log_result(
                            "TTS",
                            "Additional Language",
                            "ERROR",
                            f"Unexpected status code: {response.status}"
                        )
            except Exception as e:
                self.log_result(
                    "TTS",
                    "Additional Language",
                    "ERROR",
                    f"Test failed: {str(e)}"
                )


    async def run_all_tests(self):
        """Run all available tests in sequence"""
        self.logger.info("Starting complete test suite")
        
        # Network tests
        await self.run_network_tests()
        
        # Core API tests
        await self.run_core_api_tests()
        
        # TTS tests
        await self.run_tts_tests()
        
        # Print summary
        await self._print_test_summary()                    

async def main():
    tester = AllTalkTester()
    if await tester.setup():
        await tester.menu()
    else:
        print("Setup failed. Exiting.")

if __name__ == "__main__":
    asyncio.run(main())                   