import unittest
from unittest.mock import call, patch

import httpx
from groq import RateLimitError

from generation.groq_client import (
    GroqRateLimitError,
    _retry_after_seconds,
    call_with_rate_limit_retry,
)


def rate_limit_error(
    message="Please try again in 52.5ms.",
    headers=None,
):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        headers=headers,
        json={"error": {"message": message}},
    )
    return RateLimitError(
        "rate limited",
        response=response,
        body={"error": {"message": message}},
    )


class GroqRateLimitTests(unittest.TestCase):
    def test_extracts_provider_retry_delay(self):
        error = rate_limit_error()
        self.assertAlmostEqual(_retry_after_seconds(error), 0.0525)

    def test_extracts_millisecond_unit(self):
        error = rate_limit_error("Please try again in 52 milliseconds.")
        self.assertAlmostEqual(_retry_after_seconds(error), 0.052)

    def test_prefers_retry_after_header(self):
        error = rate_limit_error(
            headers={"retry-after-ms": "125"},
        )
        self.assertAlmostEqual(_retry_after_seconds(error), 0.125)

    def test_retries_short_limit_then_returns_result(self):
        error = rate_limit_error()
        operation = unittest.mock.Mock(side_effect=[error, "completed"])

        with patch("generation.groq_client.time.sleep") as sleep:
            result = call_with_rate_limit_retry(operation)

        self.assertEqual(result, "completed")
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(0.0525)

    def test_automatically_retries_six_second_provider_delay(self):
        error = rate_limit_error("Please try again in 6 seconds.")
        operation = unittest.mock.Mock(side_effect=[error, "completed"])

        with patch("generation.groq_client.time.sleep") as sleep:
            result = call_with_rate_limit_retry(operation)

        self.assertEqual(result, "completed")
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(6.0)

    def test_automatically_retries_fifty_five_second_provider_delay(self):
        error = rate_limit_error("Please try again in 55 seconds.")
        operation = unittest.mock.Mock(side_effect=[error, "completed"])

        with patch("generation.groq_client.time.sleep") as sleep:
            result = call_with_rate_limit_retry(operation)

        self.assertEqual(result, "completed")
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(55.0)

    def test_automatically_retries_275_second_provider_delay(self):
        error = rate_limit_error("Please try again in 275 seconds.")
        operation = unittest.mock.Mock(side_effect=[error, "completed"])

        with patch("generation.groq_client.time.sleep") as sleep:
            result = call_with_rate_limit_retry(operation)

        self.assertEqual(result, "completed")
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(275.0)

    def test_automatically_retries_510_second_provider_delay(self):
        error = rate_limit_error("Please try again in 510 seconds.")
        operation = unittest.mock.Mock(side_effect=[error, "completed"])

        with patch("generation.groq_client.time.sleep") as sleep:
            result = call_with_rate_limit_retry(operation)

        self.assertEqual(result, "completed")
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(510.0)

    def test_stops_after_bounded_retries_with_actionable_error(self):
        error = rate_limit_error()
        operation = unittest.mock.Mock(side_effect=error)

        with patch("generation.groq_client.time.sleep") as sleep:
            with self.assertRaises(GroqRateLimitError) as raised:
                call_with_rate_limit_retry(operation)

        self.assertEqual(operation.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(0.0525), call(0.0525)])
        self.assertIn("fewer questions", str(raised.exception))
        self.assertIn("mandatory", str(raised.exception))
        self.assertNotIn("disable optional", str(raised.exception))

    def test_does_not_sleep_for_provider_retry_window_over_ten_minutes(self):
        error = rate_limit_error("Please try again in 605 seconds.")
        operation = unittest.mock.Mock(side_effect=error)

        with patch("generation.groq_client.time.sleep") as sleep:
            with self.assertRaises(GroqRateLimitError) as raised:
                call_with_rate_limit_retry(operation)

        self.assertEqual(operation.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(raised.exception.retry_after_seconds, 605)
        self.assertIn("Wait about 605 seconds", str(raised.exception))
        self.assertIn("mandatory", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
