/*
 * Copyright (C) 2026-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#include "alternator/import.hh"

#include <boost/regex.hpp>
#include <boost/regex/v5/regex_match.hpp>
#include <memory>
#include <seastar/core/coroutine.hh>
#include "alternator/error.hh"
#include "alternator/executor.hh"
#include "alternator/executor_util.hh"
#include "audit/audit.hh"
#include "service/client_state.hh"
#include "service_permit.hh"
#include "tracing/trace_state.hh"
#include "utils/rjson.hh"

namespace alternator {

future<std::optional<executor::request_return_type>> in_progress_import_description(std::string_view client_token, rjson::value& request) {
    // FIXME: as per test_import.py::test_client_token_reuse
    // DynamoDB ignores these attributes and their children:
    // 'InputCompressionType'
    // 'InputFormat'
    // 'S3BucketSource'
    // 'TableCreationParameters'
    // So even when the above attributes differ, according to
    // DynamoDB as long as the same ClientToken is there it's
    // still the same import request.
    // FIXME: there is no persistent import metadata yet.
    // Cannot determine if the import with the same token is
    // already running.
    co_return std::nullopt;
}

// ClientToken is optional, but when given it must match the pattern DynamoDB
// documents for it: at least one character, none of which is '$'.
static bool is_valid_client_token(const std::string& client_token) {
    // ^[^\$]+$ - "ClientToken" regex.
    static const boost::regex pattern("^[^$]+$");
    return boost::regex_match(client_token, pattern);
}

std::pair<std::string, std::string> get_valid_bucket_and_prefix(const rjson::value& s3_bucket_source) {
    if (rjson::find(s3_bucket_source, "S3BucketOwner")) {
        throw api_error::validation("S3BucketOwner attribute is not supported");
    }
    // ^[a-z0-9A-Z]+[.-w]*[a-z0-9A-Z]+$ - "S3Bucket" regex, 255 max length.
    static const boost::regex pattern("^[a-z0-9A-Z]+[.-w]*[a-z0-9A-Z]+$");
    std::string s3_bucket = get_non_empty_string_attribute(s3_bucket_source, "S3Bucket");
    if (s3_bucket.length() > 255) {
        throw api_error::validation("S3Bucket attribute length must be < 256");
    }
    if (!boost::regex_match(s3_bucket, pattern)) {
        throw api_error::validation(fmt::format("S3Bucket attribute must match {} regex pattern", pattern.str()));
    }
    std::string s3_key_prefix;
    s3_key_prefix = get_string_attribute(s3_bucket_source, "S3KeyPrefix", "");
    if (s3_key_prefix.length() > 1024) {
         throw api_error::validation("S3KeyPrefix must be <= 1024");
    }
    return {std::move(s3_bucket), std::move(s3_key_prefix)};
}


future<executor::request_return_type> executor::import_table(service::client_state& client_state, tracing::trace_state_ptr trace_state, service_permit permit, rjson::value request, std::unique_ptr<audit::audit_info_alternator>& audit_info) {
    _stats.api_operations.import_table++;
    // NOTE: there is access to _proxy here it's executors field.

    // Optional parameter. Present-but-empty is rejected by the getter itself.
    auto client_token = get_non_empty_string_attribute(request, "ClientToken");
    if (!is_valid_client_token(client_token)) {
        co_return api_error::validation(
                "ClientToken attribute: value does not match the required pattern ^[^\\$]+$");
    }

    auto import_description = co_await in_progress_import_description(client_token, request);
    if (import_description) {
        co_return *import_description;
    }

    auto input_format = get_non_empty_string_attribute(request, "InputFormat");
    if (input_format != "DYNAMODB_JSON") {
        co_return api_error::validation(
                fmt::format("InputFormat attribute: must be DYNAMODB_JSON, not `{}`", input_format));
    }

    rjson::value* input_compression_type = rjson::find(request, "InputCompressionType");
    if (input_compression_type) {
        if (!(input_compression_type->IsString() && (rjson::to_string_view(*input_compression_type) == "NONE"))) {
            // FIXME: current import version only supports no compression.
            co_return api_error::validation("InputCompressionType attribute: must be NONE");
        }
    }

    rjson::value* s3_bucket_source = rjson::find(request, "S3BucketSource");
    auto [s3_bucket, s3_key_prefix] = get_valid_bucket_and_prefix(*s3_bucket_source);

    constexpr std::array unsupported_parameters{
            "S3SseAlgorithm",
            "S3SseKmsKeyId",
            "InputFormatOptions"
    };
    for (auto name : unsupported_parameters) {
        if (rjson::find(request, name)) {
            co_return api_error::validation(fmt::format("{} attribute is not supported", name));
        }
    }


    auto req_return = create_table(client_state, trace_state, permit, rjson::copy(request), audit_info);

    rjson::value response = rjson::empty_object();
    co_return rjson::print(std::move(response));
}

} // namespace alternator
