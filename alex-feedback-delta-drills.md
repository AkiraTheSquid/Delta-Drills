## Delta Drills notes

Thank you for sending this over\! The app has improved a lot since I last saw it.

Unfortunately most of my feedback here is critical. Most of this is downstream of there not being adequate explanation of how to navigate and use the app.

## Stream of consciousness while using it

Needs a lot of explanation and very little is provided.

What is the function of the knowledge graph?

What is “practice” vs “targeted practice”? I’d expect these to be two versions of the same thing but they seem to take me to very different interfaces.”

Pressing tab inside the code editor brings me to the “Run” button rather than indenting.

I can’t skip a problem, I can just say I looked up the solution. This is a justifiable design choice, but it gets in the way of my ability to explore and understand the app.

After getting a question right or wrong, it’s not clear what the consequences of the different difficulty-adjustment answers are. None of the answers presents itself as a clear “default if you’re not thinking about it” 

* Maybe there should be a “don’t downgrade / upgrade difficulty at all” option? When I get a question wrong, it might be because there was a concrete thing I didn’t know rather than it being “too hard” in some general difficulty sense.

**Going through a Colab worked example**  
[Colab link to the problem](https://colab.research.google.com/github/AkiraTheSquid/Delta-Drills/blob/main/arena-procedural-drills/prereqs_autograd_pt2/backward-func-lookup/ere/worked-02-lookup-with-fallback.ipynb#scrollTo=ex1-02)

First observation: it breaks me out of my flow to move to Colab

Some of the AI generated text is grating (to me, I’ve read way too much Claude writing)

* The example that made me write this is “This drill exercises material the flashcards can't deliver on their own — interactive tensor work in a real notebook.”  
* On the Colab page: “It is study material, not a graded drill (no completion beacon). When the steps feel obvious, move to the faded version, then the full drill.”  
  * This is a sort of over-explaining Claude does a lot. Here it’s dropping a bit of jargon (“completion beacon”) that I don’t know.  
  * I also don’t know what the “faded version” is or how to move to it.

When I get to the drill, it starts off somewhere I have no context for: “The core \`BackwardFuncLookup\` raises \`KeyError\` on a missing \`(fwd, argnum)\` key.” This only makes sense if I have recently seen something about `BackwardFuncLookup`, but I haven’t done any exercises about that yet. (Maybe my default \>0 score in this topic makes the app think I’ll know what this means?)

Since this isn’t a problem to be solved, there should probably just be a “read through and understood” button rather than “solved in target time” etc. Possibly with a “skip” option or “read through but did not understand.”

**Colab faded drill**  
I don’t like that it auto-copied something without asking me\!

The exercise is “backward-func-lookup ex2: dispatch through BackwardFuncLookup for a 2-op reverse pass” but it links me to the Colab for “backward-func-lookup — faded example 3: Complete the per-edge get\_back\_func dispatch”

I don’t know where to find my `DD_TOKEN` (it isn’t in the Account tab even though I’m logged in).

* It’s also hard to tell that I’m logged in. Aside from the “Log out” button on the account tab there is no indication of this.

I’m on Question 8 and there’s still a banner that says “Calibrating — 1 of 3  
First 3 questions use fixed difficulties to calibrate your level. The next difficulty is preset during calibration, so the usual accuracy bar is hidden until calibration finishes.”

### Individual exercise feedback: 1

There were a number of problems with this exercise:  

Let $x=$ torch.tensor(\[\[1.,2.,3.\]\], requires\_grad=True) (shape (1,3)) and $\\mathbf{y}=$ torch.tensor(\[\[1.\],\[2.\]\], requires\_grad=True) (shape $(2,1)$ ). Compute $\\mathrm{z}=(\\mathrm{x}+\\mathrm{y})$.sum(), which broadcasts x and $\\mathbf{y}$ to shape ( 2,3 ), then call $\\mathbf{z}$.backward(). PyTorch's autograd internally unbroadcasts each gradient back to the original operand shapes. Print both accumulated gradients on one line with print(x.grad.tolist(), y.grad.tolist()).

First of all, the question is hard to read and understand. It mixes code, exposition, and the question in one block.

If I understand it, it’s telling me exactly what to type: `z = (x + y).sum()` and `z.backward()`  


\`\`\`  
import torch  
x \= torch.tensor(\[\[1.0, 2.0, 3.0\]\], requires\_grad=True) \#  
( 1 , 3 )  
y \= torch.tensor(\[\[1.0\],\[2.0\]\], requires\_grad=True) \#  
( 2 , 1 )  
\# TODO: compute z \= (x \+ y).sum() then call z.backward()  
z \= None \# TODO: (x \+ y).sum()  
\# TODO: z.backward()  
print(x.grad.tolist(), y.grad.tolist())  
\`\`\`

I’m not sure whether this is the solution (after properly modifying), because it won’t run (it times out after five seconds, probably on the `import torch` step). (later note: this seems to be it, since other code blocks with import torch also time out)

### Exercise feedback 2

Same as above:  
 
Manually implement the backward pass through a sigmoid activation using numpy (no autograd). Let $x=n p . a r r a y(\[0.0,1.0,-1.0\])$ and $y=\\operatorname{sigmoid}(x)= 1 /(1+\\exp (-x))$. The local Jacobian of sigmoid is $\\mathrm{dy} / \\mathrm{dx}=\\mathrm{y} \*(1-\\mathrm{y})$. Given incoming gradient grad\_out \= np.array(\[2.0,0.5,1.0\]), apply the chain rule: $\\mathrm{dx}=$ grad\_out \* ( $\\left.\\mathrm{y}^\*(1-\\mathrm{y})\\right)$. Print the result as a list of values each rounded to $\\mathbf{4}$ decimals, using print(\[round(float(v), 4\) for v in dx\]).

It directly tells me the answers, I can mindlessly uncomment the correct code and get a passing solution (which does run this time, since there’s no `import torch`)  

