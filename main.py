# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

from src import train, predict

if __name__ == "__main__":
    # train.main()
    predict.predict()