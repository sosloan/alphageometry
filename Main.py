# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Interactive while-loop entry point for the AlphaGeometry solver.

Loads definitions and rules once, then repeatedly prompts the user for
problems to solve until they choose to quit.

Usage (DD+AR mode, no LM required):
    python Main.py

Usage (AlphaGeometry mode, requires model checkpoint):
    python Main.py --mode alphageometry \
        --ckpt_path /path/to/ckpt --vocab_path /path/to/vocab
"""

import sys

from absl import app
from absl import flags
from absl import logging
import alphageometry as ag
import ddar
import graph as gh
import lm_inference as lm
import problem as pr


_GIN_SEARCH_PATHS = flags.DEFINE_list(
    'gin_search_paths',
    ['third_party/py/meliad/transformer/configs'],
    'List of paths where the Gin config files are located.',
)
_GIN_FILE = flags.DEFINE_multi_string(
    'gin_file', ['base_htrans.gin'], 'List of Gin config files.'
)
_GIN_PARAM = flags.DEFINE_multi_string(
    'gin_param', None, 'Newline separated list of Gin parameter bindings.'
)
_MODE = flags.DEFINE_string(
    'mode', 'ddar', 'either `ddar` (DD+AR) or `alphageometry`'
)
_DEFS_FILE = flags.DEFINE_string(
    'defs_file',
    'defs.txt',
    'definitions of available constructions to state a problem.',
)
_RULES_FILE = flags.DEFINE_string(
    'rules_file', 'rules.txt', 'list of deduction rules used by DD.'
)
_CKPT_PATH = flags.DEFINE_string('ckpt_path', '', 'checkpoint of the LM model.')
_VOCAB_PATH = flags.DEFINE_string(
    'vocab_path', '', 'path to the LM vocab file.'
)
_OUT_FILE = flags.DEFINE_string(
    'out_file', '', 'path to the solution output file.'
)
_BEAM_SIZE = flags.DEFINE_integer(
    'beam_size', 1, 'beam size of the proof search.'
)
_SEARCH_DEPTH = flags.DEFINE_integer(
    'search_depth', 1, 'search depth of the proof search.'
)


def _solve(problem_txt: str, mode: str, model, out_file: str) -> None:
  """Parse problem_txt and attempt to solve it with the requested mode."""
  need_rename = mode != 'ddar'
  try:
    p = pr.Problem.from_txt(problem_txt, translate=need_rename)
  except Exception as e:  # pylint: disable=broad-except
    print(f'ERROR: could not parse problem: {e}')
    return

  if mode == 'ddar':
    g, _ = gh.Graph.build_problem(p, ag.DEFINITIONS)
    solved = ag.run_ddar(g, p, out_file)
  elif mode == 'alphageometry':
    solved = ag.run_alphageometry(
        model,
        p,
        _SEARCH_DEPTH.value,
        _BEAM_SIZE.value,
        out_file,
    )
  else:
    print(f'ERROR: unknown mode "{mode}"')
    return

  if solved:
    print('Solved successfully.')
  else:
    print('Could not solve the problem.')


def main(_):
  # Load definitions and rules once.
  ag.DEFINITIONS = pr.Definition.from_txt_file(_DEFS_FILE.value, to_dict=True)
  ag.RULES = pr.Theorem.from_txt_file(_RULES_FILE.value, to_dict=True)

  mode = _MODE.value
  if mode not in ('ddar', 'alphageometry'):
    raise ValueError(f'Unknown mode: {mode}')

  # Load language model once if needed.
  model = None
  if mode == 'alphageometry':
    if not _CKPT_PATH.value or not _VOCAB_PATH.value:
      raise ValueError(
          'alphageometry mode requires --ckpt_path and --vocab_path'
      )
    logging.info('Loading language model …')
    lm.parse_gin_configuration(
        _GIN_FILE.value, _GIN_PARAM.value, gin_paths=_GIN_SEARCH_PATHS.value
    )
    model = lm.LanguageModelInference(
        _VOCAB_PATH.value, _CKPT_PATH.value, mode='beam_search'
    )
    logging.info('Language model loaded.')

  print(f'AlphaGeometry solver — mode: {mode}')
  print('Enter a problem string (e.g. "a b c = triangle a b c ? perp a b b c")')
  print('or type "quit" / "exit" / "q" to stop.\n')

  while True:
    try:
      problem_txt = input('Problem> ').strip()
    except (EOFError, KeyboardInterrupt):
      print('\nExiting.')
      break

    if not problem_txt:
      continue

    if problem_txt.lower() in ('quit', 'exit', 'q'):
      print('Exiting.')
      break

    _solve(problem_txt, mode, model, _OUT_FILE.value)
    print()


if __name__ == '__main__':
  app.run(main)
