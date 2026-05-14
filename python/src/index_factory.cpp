// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Translation unit that owns the out-of-line definition of IndexFactory::create.
// Keeping the dispatch tree (33 PyIndex<Builder, Space> specializations) in a
// single .o lets the rest of the binding (notably pybind.cpp) skip re-instantiating
// those templates. The generated header pulls in index.hpp itself, so include
// order is self-managed and immune to alphabetical reordering by clang-format.

#include "dispatch_generated.hpp"
