/*
 * Copyright (C) 2026-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#pragma once

#include <optional>
#include <string_view>
#include "alternator/executor.hh"
#include "utils/rjson.hh"

namespace alternator {


// FIXME: stub. Imports are not persisted yet, so there is nothing to look the token up.
// Once import state is stored, this needs to consult it and honour the 8-hour window and
// desribe the existing import or return an error when parameters don't match.
future<std::optional<executor::request_return_type>> describe_existing_import(std::string_view client_token, rjson::value& request);

} // namespace alternator
