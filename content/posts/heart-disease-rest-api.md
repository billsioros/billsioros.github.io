---
title: "Building a REST API"
date: 2024-09-05
draft: false
colab: "https://colab.research.google.com/github/billsioros/billsioros.github.io/blob/master/static/code/heart-disease-rest-api.ipynb"
author: "Vassilis Sioros"
categories:
  - Heart Disease Prediction
tags:
  - Python
  - Machine Learning
  - API
  - FastAPI
previousPost: https://billsioros.github.io/posts/heart-disease-prediction/
image: /images/posts/heart-disease-rest-api/index.png
thumbnail: /images/posts/heart-disease-rest-api/index.thumbnail.png
description: "FastAPI to the Rescue!"
toc:
---

Heart disease is like the uninvited guest that crashes the party, causing 17.9 million deaths every year—about 31% of all deaths worldwide, with many victims under 70. But what if we could predict when this party crasher is coming? That’s where machine learning steps in, superhero-style! These models act like crystal balls for your health, scanning data for warning signs like high blood pressure or cholesterol to predict who’s at risk of heart disease.

In a [previous post](https://billsioros.github.io/posts/heart-disease-prediction/), we trained one of these models using the [Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). Now, we’re going to kick things up a notch and build a REST API to make this model accessible to the world! And our tool of choice? [`FastAPI`](https://fastapi.tiangolo.com/), *a lightning-fast, easy-to-use framework that makes building APIs with Python a breeze.*

So, what’s a [`REST`](https://en.wikipedia.org/wiki/REST) API, you ask? It’s how apps talk to each other over the internet. REST APIs handle requests like `GET` (grab info), `POST` (send info), `PUT` (update info), and `DELETE` (bye-bye info). FastAPI makes setting this up simple, fast, and fun. Ready to dive in? Let’s get started!

## Crafting a Data Blueprint

First, we need a blueprint for the data that users will send us, kind of like a form where they fill in their details. For this, we use [`Pydantic`](https://docs.pydantic.dev/latest/). So, what’s Pydantic? Think of it as the bouncer for your API—it checks that all incoming data is valid and properly structured before letting it through.

Here’s a quick rundown of what’s happening:

- This is our blueprint for incoming data. It’s like a form where users fill in their details. Each field comes with rules (e.g., age must be between 0 and 130) so we’re working with data that makes sense and fits what our machine learning model needs.
- We use `IntEnum` from Python’s `enum` module to handle categories like `Sex`, `ChestPain`, and `StSlope`. These ensure only valid options are passed.
- The `Field` function lets us set validation rules (e.g., minimum and maximum values) and add descriptions. This way, anyone using the API knows exactly what each field is for—no guesswork required!
- The `HeartBeatSchema` is an extension of `HeartBeatCreateSchema`. It adds extra fields like `id`, which acts as a unique identifier for each record, and `heart_disease`, which holds the prediction from our model. Think of `HeartBeatSchema` as mimicking a database record creation operation—it's what you'll get back once the data is processed and stored.

```python
class Sex(IntEnum):
    MALE = auto()
    FEMALE = auto()


class ChestPain(IntEnum):
    TYPICAL_ANGINA = auto()
    ATYPICAL_ANGINA = auto()
    NON_ANGINAL_PAIN = auto()
    ASYMPTOMATIC = auto()


class StSlope(IntEnum):
    UP = auto()
    FLAT = auto()
    DOWN = auto()
```

```python
class HeartBeatCreateSchema(BaseModel):
    class Config:
        from_attributes = True

    age: int = Field(..., ge=0, le=130, description="Age of the patient [years]")
    sex: Sex
    chest_pain_type: ChestPain
    fasting_blood_sugar: bool = Field(
        ...,
        description="Fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]",
    )
    max_heart_rate: int = Field(
        ...,
        ge=60,
        le=300,
        description="Maximum heart rate achieved [Numeric value between 60 and 202]",
    )
    exercise_angina: bool
    old_peak: float = Field(
        ...,
        ge=-10,
        le=10,
        description="Oldpeak = ST [Numeric value measured in depression]",
    )
    st_slope: StSlope
```

```python
class HeartBeatSchema(HeartBeatCreateSchema):
    id: str
    heart_disease: bool
```

## Building the Bot

We’ll now define a bot that predicts whether someone has heart disease based on their health data. Here’s how it functions:

1. **Initialization**: When we create our `Bot`, we supply it with the pretrained model we’ve previously developed.
2. **Data Processing**: When the bot receives a new health report in the form of a `HeartBeatCreateSchem` payload, it first converts this data into a format compatible with the model—using a pandas DataFrame, which was the format used during training.
3. **Prediction**: The bot then feeds the processed data into the model. The model evaluates the input and provides a prediction, indicating whether heart disease is likely with a `True` or `False`.


```python
class Bot(object):
    def __init__(self, model: Pipeline) -> None:
        self._model = model

    def predict(self, heartbeat: HeartBeatCreateSchema) -> bool:
        payload = {
            "Age": heartbeat.age,
            "Sex": heartbeat.sex,
            "ChestPain": heartbeat.chest_pain_type,
            "FastingBS": heartbeat.fasting_blood_sugar,
            "MaxHR": heartbeat.max_heart_rate,
            "ExerciseAngina": heartbeat.exercise_angina,
            "Oldpeak": heartbeat.old_peak,
            "ST_Slope": heartbeat.st_slope,
        }

        return self._model.predict(pd.DataFrame([payload]))[0]
```

## Loading settings from `env`

Below we use [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) to set up the configuration for our application. `pydantic-settings` is a powerful tool that simplifies managing and validating configuration settings with ease.

At the moment, our application might seem simple with just a path to our machine learning model checkpoint. However, as our project grows, we'll need to handle more complex configurations like database connection strings. In real-world applications, having a robust configuration management system is crucial for handling various settings and ensuring everything runs smoothly.

The magic happens in the `Settings` class, which is like our app’s personal assistant for configuration. It knows to read environment variables with the prefix `BACKEND_`, and ignore extra junk. As we already mentioned, we’ve only got a `checkpoint_path` that points to our model file, making sure our app knows exactly where to find it.


```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BACKEND_",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    checkpoint_path: Path = Path().cwd().parent / "data" / "model.joblib"
```

## Dependency Injection

We now define a few FastAPI dependencies. Before diving deeper into the code, let's take a fun detour into the world of [**Dependency Injection (DI)**](https://en.wikipedia.org/wiki/Dependency_injection). Dependency Injection (DI) is like having a personal assistant for your code. You tell FastAPI what your functions need, and it magically delivers those needs without you lifting a finger. It’s like asking a party planner for snacks, drinks, and music—you just specify what you want, and they handle the rest.

> In FastAPI, [`Depends`](https://fastapi.tiangolo.com/tutorial/dependencies/) is used to define dependencies.

Now, let’s look at the code:

- `get_settings`: Think of this as our settings factory, handing you a `Settings` object with all you need.
- `get_bot`: Here’s where FastAPI’s magic happens. It uses `Depends` to automatically provide the `Settings` needed to load our model and create a `Bot`.


```python
async def get_settings() -> Settings:
    return Settings()

async def get_bot(request: Request, settings: Settings = Depends(get_settings)):
    model = joblib.load(settings.checkpoint_path)

    return Bot(model)
```

## Defining the API

Let's now set up our FastAPI app with a splash of personality:

- `title="HeartBeat"`: The grand name of our app, ready to monitor those heartbeats!
- `description="A heart failure detection system"`: A quick pitch on what our app does—keeping hearts healthy.
- `version="1.0.0"`: Our app’s debut version—fresh and ready to go.
- `contact`: We’re adding a personal touch with the creator’s info, just in case anyone wants to drop a thank you note.
- `docs_url="/"`: The URL where our app’s documentation will live, making it super easy to check out.


```python
app: FastAPI = FastAPI(
    title="HeartBeat",
    description="A heart failure detection system",
    version="1.0.0",
    contact={
        "name": "Vassilis Sioros",
        "email": "billsioros97@gmail.com",
    },
    docs_url="/",
)
```

We now add a new endpoint at `/api/v1/predict` where we can send heart data and get predictions in return. Here's the scoop:

1. Our function takes heart data (`heartbeat`) and a `Bot` instance (automatically provided by FastAPI’s DI system).
2. The bot makes a prediction based on the heart data.
3. It creates a new `HeartBeatSchema` with a unique ID and the prediction result, ready to be sent back to the requester.

> The `@api.post` decorator is a neat way to tell FastAPI, *"Hey, this function should handle POST requests here!"* If you're curious about how decorators work, check out this insightful [**RealPython article**](https://realpython.com/primer-on-python-decorators/) for a deep dive.


```python
@app.post(
    "/api/v1/predict",
    response_model=HeartBeatSchema,
    status_code=status.HTTP_201_CREATED,
)
async def predict(
    heartbeat: HeartBeatCreateSchema,
    bot: Bot = Depends(get_bot),
):
    try:
        result = bot.predict(heartbeat)

        return HeartBeatSchema(
            id=str(uuid4()),
            heart_disease=result,
            **heartbeat.model_dump(),
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error.") from e
```

Alright, so here’s the deal: Running our FastAPI app in Google Colab is not as straight forward as we'd like since Colab isn’t set up for local servers. So we use a neat trick: [`nest_asyncio`](https://github.com/erdewit/nest_asyncio).

> *"By design asyncio does not allow its event loop to be nested. This presents a practical problem: When in an environment where the event loop is already running it's impossible to run tasks and wait for the result. Trying to do so will give the error `RuntimeError: This event loop is already running.`"*
>
> *"This module patches asyncio to allow nested use of asyncio.run and loop.run_until_complete."*

```python
import nest_asyncio
import uvicorn

nest_asyncio.apply()
uvicorn.run(app, port=8000)
```

We’re up and running! For running FastAPI locally on your own machine, you just need this (_No Colab magic required—just simple and direct!_):

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

We’ve dived into the exciting world of FastAPI, set up a REST API, and even integrated it with a machine learning model to build a heart failure detection system. Stay tuned for our upcoming blog posts, where we’ll explore deploying applications with Docker and Docker Compose, handling data persistence with databases, and much more :rocket: !
