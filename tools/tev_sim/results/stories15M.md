# TEV accuracy: models/stories15M.bin

dim 288, hidden 768, 6 layers, vocab 32000, int8 group size 32. Reference: 6 stories sampled from fp32 at temperature 1, 1154 positions, replayed through each backend.

## Classifier on the TEV, int8 hidden states

recall@K is how often the int8 CPU's best token lands in the TEV's top K.

| TEV classifier | recall@1 | @4 | @16 | @64 | top-p nucleus in top 64 |
|---|---|---|---|---|---|
| blend:16:0.25 | 74.0% | 95.4% | 99.8% | 100.0% | 95.1% |
| clamp8:16:0.5 | 91.8% | 99.9% | 100.0% | 100.0% | 97.8% |
| exact11:8:1.0 | 99.1% | 100.0% | 100.0% | 100.0% | 98.0% |

## End to end, teacher-forced against fp32

| backend | top-1 vs fp32 | KL (nats) | ppl | TEV saturation |
|---|---|---|---|---|
| fp32 CPU | 100.0% | 0.0000 | 2.77 | - |
| int8 CPU (the Wii today) | 98.4% | 0.0005 | 2.77 | - |
| split (blend/16 cols, headroom 0.25, CPU top-256) | 98.4% | 0.0046 | 2.78 | cls 7.19% |
| GPU only (clamp8/16 cols, headroom 0.5), blended classifier + CPU top-256 | 90.6% | 0.0403 | 2.88 | wq 1.82%, wk 1.82%, wv 1.60%, wo 1.73%, w1 1.82%, w2 0.48%, w3 1.80%, cls 6.75% |
| GPU only (exact11/8 cols, headroom 1.0), blended classifier + CPU top-256 | 97.1% | 0.0057 | 2.78 | wq 0.00%, wk 0.00%, wv 0.00%, wo 0.00%, w1 0.00%, w2 0.00%, w3 0.00%, cls 7.12% |
| split (clamp8/16 cols, headroom 0.5, CPU top-64) | 98.4% | 0.0008 | 2.77 | cls 1.23% |
| GPU only (clamp8/16 cols, headroom 0.5) | 87.3% | 0.0608 | 2.97 | wq 1.82%, wk 1.82%, wv 1.60%, wo 1.73%, w1 1.82%, w2 0.48%, w3 1.80%, cls 1.22% |
| GPU only (exact11/8 cols, headroom 1.0) | 96.9% | 0.0022 | 2.77 | wq 0.00%, wk 0.00%, wv 0.00%, wo 0.00%, w1 0.00%, w2 0.00%, w3 0.00%, cls 0.00% |

## Greedy stories

### "Once upon a time"

**fp32 CPU**

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
>
> Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.
>
> Lily found a stick and tried to hit the ball. But the stick was

**int8 CPU (the Wii today)**

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
>
> Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to get the ball down.
>
> Lily found a long stick and tried to reach the ball. She poked

**split (blend/16 cols, headroom 0.25, CPU top-256)** (same as int8 CPU)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
>
> Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to get the ball down.
>
> Lily found a long stick and tried to reach the ball. She poked

**GPU only (clamp8/16 cols, headroom 0.5), blended classifier + CPU top-256** (diverges from int8 CPU at token 34)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, dark cloud in the sky. She knew that meant it might rain soon.
>
> Lily went inside her house and saw her mommy cooking in the kitchen. "Mommy, can I help you cook?" Lily asked. "Sure, sweetie," her mommy replied.
>
> Lily helped her mommy make a yummy soup. They put the soup in a pot and waited for it to cook.

**GPU only (exact11/8 cols, headroom 1.0), blended classifier + CPU top-256** (diverges from int8 CPU at token 89)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
>
> Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she saw a big, strong man walking by. He was very reliable and always helped her when she needed it.
>
> Lily asked the man if he could help

**split (clamp8/16 cols, headroom 0.5, CPU top-64)** (same as int8 CPU)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
>
> Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to get the ball down.
>
> Lily found a long stick and tried to reach the ball. She poked

**GPU only (clamp8/16 cols, headroom 0.5)** (diverges from int8 CPU at token 34)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, dark cloud in the sky. It looked like a big, fluffy pillow. Lily knew that meant it might rain soon.
>
> Lily's mommy told her to come inside before it started to rain. Lily didn't want to go inside, but she knew her mommy was right. She went inside and watched the rain. It was wet outside, but she was happy to be inside.

**GPU only (exact11/8 cols, headroom 1.0)** (diverges from int8 CPU at token 76)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
>
> Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump, but she couldn't reach it. She felt sad.
>
> Then, Lily had an idea. She found a big box and put it under the ball. She climbed on the box and reached up high.

### "The grumpy turtle king"

**fp32 CPU**

> The grumpy turtle king was walking in the forest. He was carrying a big bag of food. Suddenly, he heard a loud noise. He looked up and saw a big bird flying above him. The bird was carrying a big bag of food.
>
> The king was very angry. He shouted at the bird and said, "That's my food! Give it back!" The bird just flew away. The king was so angry that he started to cry.
>
> The king's father came to the forest and saw the king crying. He asked the

**int8 CPU (the Wii today)**

> The grumpy turtle king was walking in the forest. He was carrying a big bag of food. Suddenly, he heard a loud noise. He looked up and saw a big bird flying above him. The bird was carrying a big bag of food.
>
> The king was very angry. He shouted at the bird and said, "That's my food! Give it back!" The bird just flew away. The king was so angry that he started to cry.
>
> The king's friend, a little rabbit, came to see what was wrong. The

**split (blend/16 cols, headroom 0.25, CPU top-256)** (same as int8 CPU)

> The grumpy turtle king was walking in the forest. He was carrying a big bag of food. Suddenly, he heard a loud noise. He looked up and saw a big bird flying above him. The bird was carrying a big bag of food.
>
> The king was very angry. He shouted at the bird and said, "That's my food! Give it back!" The bird just flew away. The king was so angry that he started to cry.
>
> The king's friend, a little rabbit, came to see what was wrong. The

**GPU only (clamp8/16 cols, headroom 0.5), blended classifier + CPU top-256** (diverges from int8 CPU at token 22)

> The grumpy turtle king was walking in the forest. He was carrying a big bag of treasure. Suddenly, he heard a voice.
>
> "What are you doing?" asked the voice.
>
> The king looked around and saw a little girl. She was only three years old.
>
> "I'm looking for treasure," the king said.
>
> The little girl smiled. "I know where to find it. Follow me!"
>
> The king followed the little girl and they walked through the forest. The little girl pointed to a tree and said, "Look,

**GPU only (exact11/8 cols, headroom 1.0), blended classifier + CPU top-256** (diverges from int8 CPU at token 11)

> The grumpy turtle king was walking along the beach. He was looking for something to do. Suddenly, he saw a little girl. She was playing in the sand and was making a sand castle.
>
> The king said to the little girl, "What are you doing?"
>
> The little girl said, "I'm making a castle!"
>
> The king said, "That's very nice. Can I help you?"
>
> The little girl said, "Yes, please!"
>
> So the king and the little girl worked together to make the castle. They used st

**split (clamp8/16 cols, headroom 0.5, CPU top-64)** (same as int8 CPU)

> The grumpy turtle king was walking in the forest. He was carrying a big bag of food. Suddenly, he heard a loud noise. He looked up and saw a big bird flying above him. The bird was carrying a big bag of food.
>
> The king was very angry. He shouted at the bird and said, "That's my food! Give it back!" The bird just flew away. The king was so angry that he started to cry.
>
> The king's friend, a little rabbit, came to see what was wrong. The

**GPU only (clamp8/16 cols, headroom 0.5)** (diverges from int8 CPU at token 11)

> The grumpy turtle king was walking along the beach. He was looking for something special. He saw a big, shiny rock and he decided to take it.
>
> The king said, "This rock is mine!" He was so happy. He took the rock and put it in his pocket.
>
> The king was so happy with his new rock. He said, "This rock is mine now!" He took the rock and went back to his castle.
>
> The king was so happy with his new rock. He put it on the table and went to sleep. The next morning

**GPU only (exact11/8 cols, headroom 1.0)** (diverges from int8 CPU at token 11)

> The grumpy turtle king was walking along the beach. He was looking for something to do. Suddenly, he saw a little girl. She was playing in the sand and was making a sand castle. The king was so excited that he started to clap his hands.
>
> The little girl was so happy that she started to clap too. She was so happy that she started to dance. The king was so surprised that he stopped clapping and started to laugh.
>
> The little girl was so happy that she started to clap too. She kept on clapping and
