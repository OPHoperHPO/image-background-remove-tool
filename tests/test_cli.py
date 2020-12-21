"""
Name: tests
Description: This file contains the test code
Version: [release][3.2]
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Anodev (OPHoperHPO)[https://github.com/OPHoperHPO] .
License: Apache License 2.0
License:
   Copyright 2020 OPHoperHPO

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import subprocess
import unittest


def cli_call_old(input, out, model):
    sub = subprocess.Popen("python3 ../main.py -i {} -o {} -m {}".format(input, out, model), shell=True,
                           stdout=subprocess.PIPE)
    return sub.communicate()[0].decode("UTF-8").replace('\n', '')


def cli_call(input, out, model, prep="bla", post="bla"):
    sub = subprocess.Popen("python3 ../main.py -i {} -o {} -m {} -prep {} -postp {}".format(input, out, model, prep,
                                                                                            post), shell=True,
                           stdout=subprocess.PIPE)
    return sub.communicate()[0].decode("UTF-8").replace('\n', '')


class CliTest(unittest.TestCase):
    def test_cli(self):
        self.assertEqual(cli_call_old("docs/imgs/input/1.jpg", "docs/imgs/examples/u2netp/", "test"),
                         "docs/imgs/input/1.jpg docs/imgs/examples/u2netp/ test bbd-fastrcnn rtb-bnb")
        self.assertEqual(cli_call_old("docs/imgs/input/1.jpg", "docs/imgs/examples/u2netp/1.png", "test"),
                         "docs/imgs/input/1.jpg docs/imgs/examples/u2netp/1.png test bbd-fastrcnn rtb-bnb")
        self.assertEqual(cli_call_old("docs/imgs/input/", "docs/imgs/examples/u2netp/", "test"),
                         "docs/imgs/input/ docs/imgs/examples/u2netp/ test bbd-fastrcnn rtb-bnb")
        self.assertEqual(cli_call("docs/imgs/input/1.jpg", "docs/imgs/examples/u2netp/", "test", "BLA-BAL", "daw"),
                         "docs/imgs/input/1.jpg docs/imgs/examples/u2netp/ test BLA-BAL daw")
        self.assertEqual(cli_call("docs/imgs/input/1.jpg", "docs/imgs/examples/u2netp/", "test", "dawdwa", "daw"),
                         "docs/imgs/input/1.jpg docs/imgs/examples/u2netp/ test dawdwa daw")


if __name__ == '__main__':
    unittest.main()
