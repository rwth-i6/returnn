Please always report problems by filling an issue here on GitHub.
This also covers reporting problems with the documentation (e.g. if sth is unclear).

Or you could try to improve the code, or extend the code, or the documentation.

General rules when contributing to the code of RETURNN:

* The master branch of RETURNN is considered as stable.
  Keep it that way.
  (This is for everything which is used by other people.
   If you work on some new experimental feature, which would not affect other people,
   or other people are well aware of the current development on it,
   it's okay to push incomplete or experimental code to the master branch.
   It is even encouraged to commit often and early and not wait too long.)
* Do not break existing setups (unless there is a very good reason to do so).
* Keep consistent and clean code style.
  Our code style follows mostly PEP8, except for 2 space indents, and 120 char line limits.
  Our code style is extended by a lot of common Python conventions.
  (If you are not an expert in Python, you will likely not know about PEP8 or standard Python conventions.
   In that case, I very much recommend to use an IDE like PyCharm which will help you keeping these standards.)
* Make sure all [tests](https://returnn.readthedocs.io/en/latest/advanced/test_suite.html) pass.
* At the time being, we want to support earlier versions of TF 1
  (consider at least TF 1.8, but maybe even TF 1.4)
  and also the most recent TF versions.
  So do not break existing setups.
  However, for new features, it is ok to target TF >=1.14
  (which already provides `tf.compat`),
  and in general to use newer TF features (even maybe TF >=2),
  as long as your new feature (new layer or so) is just optional.
  For older TF support, `TFCompat.py` might be helpful. (See code for examples.)

About new features:

* Is this useful for someone else? If not, or not sure yet, just put this into your config.
  Almost all changes / extensions can be put into the config, without modifying RETURNN code.
  (And if this is not possible for your specific change yet, or too complicated,
   then let's discuss how to extend or generalize RETURNN
   such that RETURNN becomes generic enough for this,
   such that you do not need to modify RETURNN.)
* Write them in a generic way, that is easily composable,
  and reflects a core atomic functionality or concept,
  and not too much at once.
  The (class/function/layer/whatever) name should reflect
  the functionality/concept,
  and less a specific task/model.
  (Good examples: `LinearLayer`, `ConvLayer`, `DotLayer`, `EvalLayer`,  ...;
   bad examples: `AllophoneStateIdxParserLayer`, `NeuralTransducerLayer`,
   `TranslationDataset`, `LmDataset`).
  It should be easy to write a config using this,
  but also easy to understand what it does when reading a config.
  If you read a config, see `"class": "linear"`,
  it should be clear just from reading the config what is happening there,
  without needing to look up the RETURNN documentation or code
  (assuming a certain minimal amount of familiarity with RETURNN).
  Functions/classes/layers should ideally have not much arguments, as a consequence of this.
  (If you keep adding options to your class, then you likely have not followed this principle
   of simplicity, and your class does too much at once.
   Better redesign it to have it atomic and move the flexibility and variations to the config.)
* If this is not going to be used by everyone,
  you (as a user of this part of the code)
  are responsible for this part of the code.
  This also means that you should have written tests such that other people will not accidentally break this.
  This is your responsibility.
* Even if this is work-in-progress, incomplete or experimental,
  directly push this to the master branch
  (or prepare for master branch -- see below on pull requests).
  But write it in a way that it will not affect other parts,
  i.e. that everything else keeps being stable,
  and this would be an optional feature.

The common process of a change would be like:

* Make sure you read [the documentation](https://returnn.readthedocs.io/).
  And also watched the introduction video.
* If you are new to the project, make a pull request.
* If you are unsure about some change / extension
  (e.g. how to implement it, or how to fix it),
  talk to some of us first
  (e.g. our internal Slack RETURNN channel, or via mail),
  or make a pull request, or an issue.
* Only if you are very sure about your change, and it is not too big,
  or only likely used by yourself (for now) and cannot possibly break anyones else setup,
  then you can directly commit it.
  (Always check the tests.)

