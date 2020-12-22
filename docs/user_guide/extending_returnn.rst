.. _extending_returnn:

=================
Extending RETURNN
=================

First of all:

Is this extension or modification
useful for someone else, now and also definitely in the upcoming 5 years?
If not, or not sure yet, just put this into your config.
Almost all changes / extensions can be put into the config,
without modifying RETURNN code.
(And if this is not possible for your specific change yet, or too complicated,
then let's discuss how to extend or generalize RETURNN
such that RETURNN becomes generic enough for this,
such that you do not need to modify RETURNN.)

**Definitely do not touch the existing RETURNN code if you don't plan to push this back to the master!**
This goes against the whole idea of scientific research.
Scientific research, if anything useful comes out of it, should be reusable by others.
Once you start hacking around in your RETURNN fork,
you make it very hard that others can reuse your work.
So, either right from the beginning implement this in a generic way
which is definitely useful for others (see point above),
and then this can be pushed to the master.
Or just add this to your config.
If it turns out to be useful, we can then add it later.

If unsure, before implementing anything, just talk to some of us!

Now, having said that, any contribution is very welcome!

Read `this <https://github.com/rwth-i6/returnn/blob/master/CONTRIBUTING.md>`__
about how to work with the code, the code style, general rules, etc.

If this is a more radical change or new feature,
this definitely should be discussed first with us,
about how this should be implemented.
Open a GitHub issue for that.

If this will be a bug fix, it is already helpful to reduce this to a simple test case.
The usual procedure would be to create a new branch having your simple test case which triggers the bug,
and then create a PR (in draft mode) for this branch.
This allows for collaborative work on a fix.

Also see :ref:`recent_development`.

Also see other `merged pull requests <https://github.com/rwth-i6/returnn/pulls?q=is%3Apr+is%3Amerged+>`__.
