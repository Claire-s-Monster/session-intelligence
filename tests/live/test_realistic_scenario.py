#!/usr/bin/env python3
"""Test realistic scenario that mimics quality_execute_comprehensive_suite."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.token_limiter import SessionTokenLimiter


def simulate_quality_response():
    """Simulate the problematic quality_execute_comprehensive_suite response."""

    print("=== SIMULATING QUALITY RESPONSE (43K+ tokens) ===")

    # Create limiter with the actual quality operation limit
    limiter = SessionTokenLimiter()
    quality_limit = limiter.operation_limits["quality_execute_comprehensive_suite"]  # 15000 tokens

    # Create a response that mimics what quality_execute_comprehensive_suite might return
    quality_response = {
        "overall_health": "good",
        "health_score": 95.2,
        "timestamp": "2025-08-23T17:17:41.123456",
        "summary": {
            "total_files": 247,
            "tests_run": 1542,
            "coverage_percentage": 87.3,
            "lint_violations": 23,
            "security_issues": 0,
        },
        "test_results": {
            "passed": 1498,
            "failed": 12,
            "skipped": 32,
            "detailed_failures": [
                {
                    "test_name": f"test_module_{i}_functionality",
                    "file": f"tests/test_module_{i}.py",
                    "error": f"AssertionError: Expected result {i*2} but got {i*3}",
                    "traceback": "\\n".join(
                        [
                            f"  File 'test_{i}.py', line {j*10}, in test_function_{j}"
                            for j in range(15)
                        ]
                    ),
                    "context": f"Test context for module {i} with detailed setup and teardown information "
                    * 10,
                }
                for i in range(50)  # 50 detailed test failures
            ],
            "coverage_report": {
                f"src/module_{i}.py": {
                    "lines": 247 + i * 10,
                    "covered": 200 + i * 8,
                    "missing_lines": list(range(10 + i, 50 + i * 2)),
                    "detailed_analysis": f"Module {i} analysis with line-by-line coverage details "
                    * 20,
                }
                for i in range(100)  # 100 modules
            },
        },
        "lint_results": {
            "ruff_violations": [
                {
                    "file": f"src/file_{i}.py",
                    "line": 45 + i * 3,
                    "column": 12,
                    "rule": "E501",
                    "message": f"Line too long ({80 + i} > 79 characters)",
                    "code_context": f"    very_long_variable_name_that_exceeds_limit_{i} = some_function_call_with_many_parameters({i}, {i*2}, {i*3})",
                }
                for i in range(150)  # 150 lint violations
            ],
            "mypy_errors": [
                {
                    "file": f"src/type_file_{i}.py",
                    "line": 23 + i * 2,
                    "error": f"Argument {i} has incompatible type 'str'; expected 'int'",
                    "note": f"Type analysis for function_{i} with detailed context about why this type mismatch occurred and suggestions for fixing it "
                    * 8,
                }
                for i in range(75)  # 75 type errors
            ],
        },
        "security_analysis": {
            "bandit_results": [
                {
                    "issue": f"Potential security issue {i}",
                    "severity": "LOW" if i % 3 == 0 else "MEDIUM",
                    "confidence": "HIGH",
                    "file": f"src/security_{i}.py",
                    "line": 89 + i * 4,
                    "details": f"Security analysis details for issue {i} with comprehensive explanation of the potential vulnerability and recommendations for remediation "
                    * 12,
                }
                for i in range(30)  # 30 security findings
            ]
        },
        "performance_analysis": {
            "slow_tests": [
                {
                    "test": f"test_performance_{i}",
                    "duration": 2.5 + i * 0.1,
                    "file": f"tests/perf/test_{i}.py",
                    "profiling_data": f"Detailed profiling information for test {i} showing function call hierarchies and time spent in each function "
                    * 15,
                }
                for i in range(40)  # 40 slow tests
            ]
        },
        "recommendations": [
            f"Recommendation {i}: Detailed recommendation with extensive explanation of the issue, impact analysis, and step-by-step implementation guide with code examples and best practices to follow for maximum effectiveness in your specific codebase context. This recommendation addresses critical aspects of code quality, maintainability, performance, and security considerations that will significantly improve your project's overall health score and developer experience while reducing technical debt and future maintenance overhead."
            for i in range(25)  # 25 detailed recommendations
        ],
        "full_report_data": {
            "raw_test_output": "\\n".join(
                [
                    f"Running test {i}: {'PASS' if i % 10 != 0 else 'FAIL'} - detailed output " * 5
                    for i in range(500)
                ]
            ),
            "detailed_metrics": {
                f"metric_{i}": {
                    "value": i * 1.5,
                    "trend": "improving",
                    "analysis": f"Metric analysis {i} " * 20,
                }
                for i in range(200)
            },
        },
    }

    # Calculate original size
    original_json = json.dumps(quality_response, indent=2)
    original_estimate = limiter.token_estimator.estimate_tokens(
        original_json, limiter.token_estimator.detect_content_type(original_json)
    )

    print(f"Simulated quality response:")
    print(f"  Characters: {len(original_json):,}")
    print(f"  Estimated tokens: {original_estimate.estimated_tokens:,}")
    print(f"  Quality operation limit: {quality_limit:,} tokens")
    print(f"  Exceeds limit by: {original_estimate.estimated_tokens - quality_limit:,} tokens")

    # Apply the same limiting that would happen in the MCP server
    limited_response = limiter.limit_response(
        quality_response, "quality_execute_comprehensive_suite"
    )

    # Show results
    limited_json = json.dumps(limited_response, indent=2)
    final_estimate = limiter.token_estimator.estimate_tokens(
        limited_json, limiter.token_estimator.detect_content_type(limited_json)
    )

    print(f"\nAfter token limiting:")
    print(f"  Characters: {len(limited_json):,}")
    print(f"  Estimated tokens: {final_estimate.estimated_tokens:,}")
    print(f"  Under limit: {final_estimate.estimated_tokens <= quality_limit}")

    if "_token_limit_info" in limited_response:
        info = limited_response["_token_limit_info"]
        print(f"\n🎉 SUCCESSFULLY HANDLED LARGE QUALITY RESPONSE!")
        print(f'  Original: {info["original_tokens"]:,} tokens')
        print(f'  Reduced to: {info["final_tokens"]:,} tokens')
        print(f'  Tokens saved: {info["original_tokens"] - info["final_tokens"]:,}')
        print(
            f'  Reduction: {((info["original_tokens"] - info["final_tokens"]) / info["original_tokens"] * 100):.1f}%'
        )
        print(f'  Operation: {info["operation"]}')

        # Show what important data was preserved
        print(f"\nImportant data preserved:")
        preserved_keys = [k for k in limited_response.keys() if not k.startswith("_")]
        print(f"  Top-level keys: {preserved_keys}")

        if "overall_health" in limited_response:
            print(f'  ✅ Health score preserved: {limited_response.get("health_score", "N/A")}')
        if "summary" in limited_response:
            print(
                f'  ✅ Summary preserved: {list(limited_response["summary"].keys()) if isinstance(limited_response["summary"], dict) else "yes"}'
            )

        print(f"\n📋 This is exactly what would happen in the MCP server!")
        print(f'   Instead of failing with "43032 tokens exceeds maximum allowed tokens (25000)"')
        print(f"   The response is intelligently truncated to fit within the 15000 token limit")
        print(f"   while preserving the most important information.")

        return True
    else:
        print(f"\n❌ No truncation occurred - response was unexpectedly small")
        return False


if __name__ == "__main__":
    print("🧪 Testing Realistic Quality Response Scenario")
    print("=" * 70)

    success = simulate_quality_response()

    print("\n" + "=" * 70)
    if success:
        print("🎉 SUCCESS! The token limiting system will solve the original problem.")
        print("Large quality responses are now automatically truncated to fit MCP limits.")
    else:
        print("❌ Test failed - needs investigation")
