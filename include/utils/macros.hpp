/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * @file macros.hpp
 * @brief Common macros for special member function declarations.
 *
 * This file provides macros to simplify the declaration of special member
 * functions (copy/move constructors and assignment operators).
 *
 * Usage:
 * @code
 *   class MyClass {
 *    public:
 *     ALAYA_NON_COPYABLE_BUT_MOVABLE(MyClass)
 *     // ... rest of the class
 *   };
 * @endcode
 */

/**
 * @brief Disable copy constructor and copy assignment operator.
 *
 * Use this when a class should not be copyable (e.g., owns unique resources).
 */
#define ALAYA_NON_COPYABLE(ClassName)    \
  ClassName(const ClassName &) = delete; \
  auto operator=(const ClassName &)->ClassName & = delete

/**
 * @brief Enable default move constructor and move assignment operator.
 *
 * Use this to explicitly enable move semantics with default implementation.
 * Move operations are marked noexcept to enable optimizations in STL containers.
 */
#define ALAYA_DEFAULT_MOVABLE(ClassName)      \
  ClassName(ClassName &&) noexcept = default; \
  auto operator=(ClassName &&) noexcept -> ClassName & = default

/**
 * @brief Disable copy but enable default move semantics.
 *
 * This is the most common pattern for classes that own unique resources
 * but should still be movable (e.g., classes with unique_ptr members).
 */
#define ALAYA_NON_COPYABLE_BUT_MOVABLE(ClassName) \
  ALAYA_NON_COPYABLE(ClassName);                  \
  ALAYA_DEFAULT_MOVABLE(ClassName)

/**
 * @brief Disable both copy and move operations.
 *
 * Use this for singleton-like classes or classes that should never be
 * copied or moved (e.g., thread-local resources, lock guards).
 */
#define ALAYA_NON_COPYABLE_NON_MOVABLE(ClassName)          \
  ClassName(const ClassName &) = delete;                   \
  auto operator=(const ClassName &)->ClassName & = delete; \
  ClassName(ClassName &&) = delete;                        \
  auto operator=(ClassName &&)->ClassName & = delete
