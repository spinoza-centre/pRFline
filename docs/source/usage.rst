.. include:: links.rst

------------
Usage notes
------------

This is a list:

1) Element 1
2) Element 2
3) Element 3


.. code:: bash

    # this is bash code
    declare -a HELLO=("hello" "world")
    for word in ${HELLO[@]}; do
        echo $word
    done

.. code:: python

    # this is python code
    HELLO = ['hello', 'world']
    for ii in HELLO:
        print(ii)

.. note:: 
   This is one of those blue note-blocks you see in documentation

.. attention:: 
   This is one of those cool attention blocks.

.. tip:: 
   This is a nice **TIP** box.

Using a *links.rst* file, I can refer to fMRIprep_'s website. Note that in the link file, the name is preceded by an underscore (e.g., '_fMRIprep'), and when we want to refer to it here we put the underscore at the end: fMRIprep_