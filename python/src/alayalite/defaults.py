# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module defines and exposes default configuration parameters for the AlayaLite library,
such as default values for index build and search parameters.
"""

from ._alayalitepy import defaults as _defaults

R = _defaults.R
L = _defaults.L
M = _defaults.M
NUM_THREADS = _defaults.NUM_THREADS
EF = _defaults.EF
