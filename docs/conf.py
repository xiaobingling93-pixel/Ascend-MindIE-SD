#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Sphinx configuration for MindIE SD documentation."""

PROJECT = "MindIE SD"
COPYRIGHT_TEXT = "2024-2026, Huawei Technologies Co., Ltd."
AUTHOR = "Huawei Technologies Co., Ltd."
EXTENSIONS = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

SOURCE_SUFFIX = {
    ".md": "markdown",
}

EXCLUDE_PATTERNS = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

TEMPLATES_PATH: list[str] = []
HTML_STATIC_PATH: list[str] = []

MYST_ENABLE_EXTENSIONS = [
    "colon_fence",
    "deflist",
]
MYST_HEADING_ANCHORS = 3
SUPPRESS_WARNINGS = [
    "image.not_readable",
    "myst.header",
    "myst.xref_missing",
    "toc.not_included",
    "toc.not_readable",
]

HTML_THEME = "sphinx_book_theme"
HTML_TITLE = "MindIE SD Documentation"
HTML_THEME_OPTIONS = {
    "repository_url": "https://gitcode.com/Ascend/MindIE-SD",
    "repository_provider": "gitlab",
    "use_repository_button": True,
    "path_to_docs": "docs",
}

globals().update(
    {
        "project": PROJECT,
        "copyright": COPYRIGHT_TEXT,
        "author": AUTHOR,
        "extensions": EXTENSIONS,
        "source_suffix": SOURCE_SUFFIX,
        "exclude_patterns": EXCLUDE_PATTERNS,
        "templates_path": TEMPLATES_PATH,
        "html_static_path": HTML_STATIC_PATH,
        "myst_enable_extensions": MYST_ENABLE_EXTENSIONS,
        "myst_heading_anchors": MYST_HEADING_ANCHORS,
        "suppress_warnings": SUPPRESS_WARNINGS,
        "html_theme": HTML_THEME,
        "html_title": HTML_TITLE,
        "html_theme_options": HTML_THEME_OPTIONS,
    }
)
